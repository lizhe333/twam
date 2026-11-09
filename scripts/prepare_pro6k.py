#!/usr/bin/env python3
"""Prepare an isolated Isaac 5.1 / CUDA 12.8 runtime, without launching simulation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import urllib.parse
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from pro6k_contract import PROFILE, UNIVTAC_COMMIT, CONTACT_RESET_PATCH

ROOT = Path(__file__).resolve().parents[2]
ENV_NAME = 'pro6k-isaac51-transfer-v1'
GRAPHICS_DIR = 'graphics-glvnd-v2'
GLVND_BUILD = '1.7.0-ha4b6fd6_5'
# GL/EGL frontends and their dispatch table must come from the same build.
GLVND_PACKAGES = {
    'libglvnd': ('6eb77601a44b78c4631d886d54ef54f321c2bdc69fdefc4065a1e0491f030c76', ('libGLdispatch.so.0',)),
    'libopengl': ('7bdca717c840e17d7dd050f925dd964da61086c2612b8579a4ff4147105f291f', ('libOpenGL.so.0',)),
    'libegl': ('13d3fde9c1dcf254968cc70652222a43f25d04e69555ccf855553110e753288e', ('libEGL.so.1',)),
    'libgl': ('7b2e7e31e8a4f5a01c45ecadcd8aa68c8f733b035240bc7ba9881835ef4bf7b4', ('libGL.so.1',)),
    'libglx': ('d1f0d51aea6bcc5d915969cbed32e72a8ed6a95931b87357ef8d0866573688c4', ('libGLX.so.0',)),
    'libgles': ('ec0854956418ca7e486ae9e74ca7a00798ca447f2db9c3f4fbe719f9f76225dc', ('libGLESv1_CM.so.1', 'libGLESv2.so.2')),
}
GRAPHICS_SONAMES = tuple(name for _, names in GLVND_PACKAGES.values() for name in names)
NATIVE_ARCHITECTURES = ('sm_89', 'sm_120')
CONTACT_SOURCE = ('third_party/TacEx/source/tacex_uipc/libuipc/'
                  'src/backends/cuda/engine/sim_engine_do_advance.cu')
CONTACT_ORIGINAL_SHA256 = '319ecca366daaa4f5e1d1f82946452cf8ba8d123c634ad3692efa232f4d879b8'
CONTACT_PATCHED_SHA256 = '8a6624943415ea483d8e6857af91db7c4b283940f4f16ddbdd5a2a00909d25bf'
SCATTER_URL = ('https://files.pythonhosted.org/packages/f5/ab/'
               '2a44ecac0f891dd0d765fc59ac8d277c6283a31907626560e72685df2ed6/torch_scatter-2.1.2.tar.gz')
SCATTER_SHA256 = '69b3aa435f2424ac6a1bfb6ff702da6eb73b33ca0db38fb26989c74159258e47'
VCPKG_COMMIT = 'dd3097e305afa53f7b4312371f62058d2e665320'
CUROBO_COMMIT = '0a50de1ba72db304195d59d9d0b1ed269696047f'
TINYGltf_ARCHIVE_SHA512 = ('c51d3e09d2980a9357c2c51598f71328c003bd473486393668bbf3f996d92b93'
                            'd4a544c42f32ca94233f8cddaf80bfaa6864967498081ade5c1d7547b35aec14')
TINYGltf_OVERLAY_SHA256 = {
    'portfile.cmake': '5c3234f821080c1daf048592a3bc110c372ec2e5eb9920af8a2dac449ec7ddd0',
    'vcpkg.json': '15b5e93b460f99d71d909ab3247df1d72f001fa05738c1149ea14fcfa80803fe',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def run(command, *, env=None, cwd=None):
    command = list(map(str, command))
    print('PREPARE ' + ' '.join(command), flush=True)
    subprocess.run(command, check=True, env=env, cwd=cwd)


def download(url, destination, expected=None):
    destination = Path(destination)
    if destination.exists() and (expected is None or digest(destination) == expected):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.download')
    print(f'DOWNLOAD {url}', flush=True)
    try:
        with urllib.request.urlopen(url, timeout=180) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output, 1024 * 1024)
        if expected is not None and digest(temporary) != expected:
            raise ValueError(f'Download checksum mismatch: {url}')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def source_inventory(source):
    folders = ('envs', 'instructions', 'task_config', 'policy', 'third_party/TacEx/source')
    return {str(p.relative_to(source)): digest(p)
            for folder in folders for p in sorted((source / folder).rglob('*'))
            if p.is_file() and (p.suffix in ('.py', '.yaml', '.yml', '.toml', '.cmake') or p.name == 'CMakeLists.txt')
            and not any(x in p.parts for x in ('build', '.git', '__pycache__'))}


def patch_source(source):
    # The source archive already includes the exact muda sources. DEV_MODE
    # avoids git-submodule operations inside a non-Git archive checkout.
    setup = source / 'third_party/TacEx/source/tacex_uipc/setup.py'
    text = setup.read_text()
    if '"-DUIPC_DEV_MODE=0"' in text:
        setup.write_text(text.replace('"-DUIPC_DEV_MODE=0"', '"-DUIPC_DEV_MODE=1"'))
    elif '"-DUIPC_DEV_MODE=1"' not in text:
        raise RuntimeError('Unknown UIPC build template')
    setup = source / 'third_party/TacEx/source/tacex/setup.py'
    text = setup.read_text()
    # Compile scatter for Ada and Blackwell instead of accepting an older prebuilt
    # CUDA extension through TacEx's direct wheel dependency.
    text, count = re.subn(r'\(\s*#[^\n]*\n\s*"torch_scatter @"\s*"https://data\.pyg\.org/[^\"]+"\s*\),',
                          '"torch_scatter==2.1.2",', text)
    if count != 1 and '"torch_scatter==2.1.2"' not in text:
        raise RuntimeError('Unknown TacEx scatter dependency template')
    setup.write_text(text)
    # An unquoted semicolon list becomes extra property arguments in CMake.
    cmake = source / 'third_party/TacEx/source/tacex_uipc/libuipc/src/backends/cuda/CMakeLists.txt'
    text = cmake.read_text()
    old, new = 'CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}', 'CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"'
    if old not in text and new not in text:
        raise RuntimeError('Unknown UIPC CUDA architecture property')
    cmake.write_text(text.replace(old, new))


def prepare_source(root, home, *, fetch):
    source = home / 'UniVTAC'
    marker = home / 'source.json'
    if not source.exists():
        partial = home / 'UniVTAC.partial'
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir()
        archive = home / 'univtac.tar'
        with archive.open('wb') as stream:
            subprocess.run(['git', '-C', str(root / 'UniVTAC'), 'archive', UNIVTAC_COMMIT,
                'envs', 'assets', 'instructions', 'task_config', 'policy/_base_policy.py',
                'third_party/TacEx/source', 'third_party/TacEx/UNIVTAC_BASELINE.md'],
                stdout=stream, check=True)
        try:
            with tarfile.open(archive) as tar:
                tar.extractall(partial, filter='data')
            partial.rename(source)
        finally:
            archive.unlink(missing_ok=True)
    elif marker.exists():
        previous = json.loads(marker.read_text())
        if previous['commit'] != UNIVTAC_COMMIT or any(
                not (source / p).is_file() or digest(source / p) != h for p, h in previous['files'].items()):
            raise RuntimeError('Prepared Isaac 5.1 source changed; use a fresh environment directory')

    patch_source(source)

    # Materialize LFS objects by their content hash. Reuse existing verified
    # workspace objects; download only absent objects into this new directory.
    missing = []
    for path in source.rglob('*'):
        if not path.is_file() or path.stat().st_size > 1024:
            continue
        blob = path.read_bytes()
        match = re.fullmatch(rb'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n?', blob)
        if match is None:
            continue
        sha, size = match[1].decode(), int(match[2])
        relative = path.relative_to(source)
        candidates = [root / '.cache/git-lfs/univtac/objects' / sha[:2] / sha[2:4] / sha,
                      root / 'UniVTAC' / relative]
        candidate = next((p for p in candidates if p.is_file() and p.stat().st_size == size and digest(p) == sha), None)
        if candidate is None and fetch:
            candidate = home / 'lfs' / sha
            name = urllib.parse.quote(relative.as_posix(), safe='/')
            download(f'https://media.githubusercontent.com/media/univtac/UniVTAC/{UNIVTAC_COMMIT}/{name}', candidate, sha)
        if candidate is None:
            missing.append(str(relative))
        else:
            path.unlink()
            try:
                os.link(candidate, path)
            except OSError:
                shutil.copy2(candidate, path)
    write(marker, dict(profile=PROFILE, commit=UNIVTAC_COMMIT,
                       files=source_inventory(source), missing_lfs=missing))
    print(f'Isaac 5.1 source ready; unresolved LFS objects: {len(missing)}', flush=True)
    return source, missing


def find_conda(home):
    configured = os.environ.get('CONDA_EXE') or shutil.which('conda')
    if configured and Path(configured).is_file():
        return Path(configured)
    prefix = home / 'miniforge'
    conda = prefix / 'bin/conda'
    if not conda.is_file():
        name = 'Miniforge3-25.3.0-3-Linux-x86_64.sh'
        url = f'https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/{name}'
        checksum = home / 'downloads' / (name + '.sha256')
        download(url + '.sha256', checksum)
        expected = checksum.read_text().split()[0]
        if not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Invalid Miniforge checksum')
        installer = home / 'downloads' / name
        download(url, installer, expected)
        run(['bash', installer, '-b', '-p', prefix])
    return conda


def prepare_graphics(root, home):
    """Prepare matching GLVND libraries without copying host NVIDIA drivers."""
    run([root / 'contact-belief-wam/.venv/bin/python',
         root / 'pi05-univtac/scripts/prepare-sim-graphics.py'],
        env=dict(os.environ, TWAM_ROOT=str(root)))
    graphics = home / GRAPHICS_DIR / 'lib'
    graphics.mkdir(parents=True, exist_ok=True)
    for source in (root / '.cache/sim-graphics-runtime/lib').glob('*.so*'):
        if source.name in GRAPHICS_SONAMES:
            continue
        temporary = graphics / (source.name + '.partial')
        shutil.copy2(source, temporary)
        temporary.replace(graphics / source.name)
    for name, (checksum, sonames) in GLVND_PACKAGES.items():
        filename = f'{name}-{GLVND_BUILD}.conda'
        package = home / 'downloads' / filename
        download(f'https://conda.anaconda.org/conda-forge/linux-64/{filename}', package, checksum)
        with zipfile.ZipFile(package) as archive:
            payload, = [n for n in archive.namelist() if n.startswith('pkg-') and n.endswith('.tar.zst')]
            unpacked = subprocess.check_output([str(home / 'sim/bin/zstd'), '-d', '-c'],
                                               input=archive.read(payload))
        with tarfile.open(fileobj=io.BytesIO(unpacked)) as archive:
            for soname in sonames:
                member, = [m for m in archive.getmembers()
                           if m.isfile() and m.name.startswith(f'lib/{soname}.')]
                temporary = graphics / (soname + '.partial')
                temporary.write_bytes(archive.extractfile(member).read())
                temporary.chmod(0o644)
                temporary.replace(graphics / soname)
    run([home / 'sim/bin/python', root / 'twam/scripts/verify_glvnd.py', '--library-dir', graphics],
        env=dict(os.environ, LD_LIBRARY_PATH=str(graphics), CUDA_VISIBLE_DEVICES='',
                 PYTHONDONTWRITEBYTECODE='1'))
    return graphics


def install(root, home, source, jobs):
    prefix = home / 'sim'
    python = prefix / 'bin/python'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
               PIP_INDEX_URL='https://pypi.org/simple', PIP_DEFAULT_TIMEOUT='120',
               PIP_RETRIES='10',
               XDG_CACHE_HOME=str(home / 'xdg-cache'),
               CONDA_PKGS_DIRS=str(root / '.cache/conda/pro6k-pkgs'),
               UV_CACHE_DIR=str(root / '.cache/uv'), OMNI_KIT_ACCEPT_EULA='YES',
               PIP_CACHE_DIR=str(root / '.cache/pip/pro6k'))
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env.pop('LD_LIBRARY_PATH', None)
    # Installation/building is CPU work; target Ada and Blackwell explicitly below.
    env['CUDA_VISIBLE_DEVICES'] = ''
    if not (home / 'toolchain.json').exists():
        conda = find_conda(home)
        run([conda, 'create' if not python.exists() else 'install', '-y', '-p', prefix,
             '--override-channels', '-c', 'conda-forge', '-c', 'nvidia',
             'python=3.11', 'pip', 'cuda-toolkit=12.8', 'cmake=3.31', 'gcc_linux-64=12',
             'gxx_linux-64=12', 'ninja', 'pkg-config', 'git', 'zip', 'unzip'], env=env)
        write(home / 'toolchain.json', dict(python='3.11', cuda='12.8', prefix=str(prefix)))
    env.update(CUDA_HOME=str(prefix), CUDA_PATH=str(prefix), CUDACXX=str(prefix / 'bin/nvcc'),
        TORCH_CUDA_ARCH_LIST='8.9;12.0', CMAKE_CUDA_ARCHITECTURES='89-real;120-real',
        CMAKE_BUILD_PARALLEL_LEVEL=str(jobs), MAX_JOBS=str(jobs), NVCC_THREADS='2',
        CC=str(prefix / 'bin/x86_64-conda-linux-gnu-gcc'),
        CXX=str(prefix / 'bin/x86_64-conda-linux-gnu-c++'),
        SETUPTOOLS_SCM_PRETEND_VERSION='0.7.7',
        PATH=str(prefix / 'bin') + os.pathsep + env.get('PATH', ''),
        LD_LIBRARY_PATH=str(prefix / 'lib'),
        CPATH=str(prefix / 'targets/x86_64-linux/include'))
    env['CUDAHOSTCXX'] = env['CXX']
    pip = [python, '-m', 'pip', 'install']
    constraints = home / 'constraints.txt'
    constraints.write_text('torch==2.7.0+cu128\ntorchvision==0.22.0+cu128\ntorchaudio==2.7.0+cu128\n'
                           'numpy<2\nisaacsim==5.1.0.0\nisaaclab==2.3.0\nwarp-lang==1.9.0\nwebsockets==12.0\n')
    env['PIP_CONSTRAINT'] = str(constraints)
    run(pip + ['setuptools==75.8.2', 'setuptools-scm==8.1.0', 'wheel==0.45.1', 'toml', 'pybind11==2.13.6', 'mypy==1.15.0'], env=env)
    run(pip + ['torch==2.7.0', 'torchvision==0.22.0', 'torchaudio==2.7.0',
               '--index-url', 'https://download.pytorch.org/whl/cu128'], env=env)
    run(pip + ['flatdict==4.0.1', '--no-build-isolation'], env=env)
    run(pip + ['isaacsim[all,extscache]==5.1.0.0', 'isaaclab==2.3.0',
               '--extra-index-url', 'https://pypi.nvidia.com'], env=env)
    env['FORCE_CUDA'] = '1'
    scatter_source = home / 'downloads/torch_scatter-2.1.2.tar.gz'
    download(SCATTER_URL, scatter_source, SCATTER_SHA256)
    run(pip + [scatter_source, '--force-reinstall', '--no-cache-dir', '--no-index',
               '--no-deps', '--no-build-isolation'], env=env)
    # Install vendored packages without their CUDA-12.6-specific wheel URL.
    run(pip + ['debugpy', 'psutil', 'nvidia-ml-py', 'wildmeshing', 'transforms3d', 'tetgen',
               'polyscope>=2.5,<3', 'decorator', 'h5py', 'omegaconf', 'imageio[ffmpeg]',
               'opencv-python<4.12', 'websockets==12.0', 'msgpack', 'dm-tree', 'numpy-quaternion',
               'networkx', 'yourdfpy', 'scikit-image', 'tqdm', 'importlib_resources'], env=env)
    tacex = source / 'third_party/TacEx/source'
    run(pip + ['--no-deps', '--no-build-isolation', '-e', tacex / 'tacex', '-e', tacex / 'tacex_assets'], env=env)
    vcpkg = home / 'vcpkg'
    if not (vcpkg / '.git').exists():
        run(['git', 'clone', '--no-checkout', 'https://github.com/microsoft/vcpkg.git', vcpkg], env=env)
        run(['git', '-C', vcpkg, 'checkout', '--detach', VCPKG_COMMIT], env=env)
    if subprocess.check_output(['git', '-C', str(vcpkg), 'rev-parse', 'HEAD'], text=True).strip() != VCPKG_COMMIT:
        raise RuntimeError('Unexpected isolated vcpkg revision')
    if not (vcpkg / 'vcpkg').exists():
        run(['bash', vcpkg / 'bootstrap-vcpkg.sh', '-disableMetrics'], env=env)
    # The project's verified 2.9.4 port satisfies UniVTAC's >=2.8.22
    # requirement. The upstream 2.9.6 archive no longer matches its pinned
    # registry hash, so reuse the already verified project source instead.
    overlay = home / 'vcpkg-overlays' / 'tinygltf'
    overlay.mkdir(parents=True, exist_ok=True)
    for name, expected in TINYGltf_OVERLAY_SHA256.items():
        original = root / '.toolchains/vcpkg-overlays/tinygltf' / name
        if not original.is_file() or digest(original) != expected:
            raise RuntimeError(f'Pinned tinygltf port changed: {original}')
        if not (overlay / name).exists():
            shutil.copy2(original, overlay / name)
        if digest(overlay / name) != expected:
            raise RuntimeError(f'Isolated tinygltf port changed: {overlay / name}')
    archive = root / '.cache/vcpkg/downloads/syoyo-tinygltf-v2.9.4.tar.gz'
    if not archive.is_file() or hashlib.sha512(archive.read_bytes()).hexdigest() != TINYGltf_ARCHIVE_SHA512:
        raise RuntimeError('Verified tinygltf 2.9.4 source archive is unavailable')
    cached_archive = vcpkg / 'downloads' / archive.name
    if not cached_archive.exists():
        try:
            os.link(archive, cached_archive)
        except OSError:
            shutil.copy2(archive, cached_archive)
    if hashlib.sha512(cached_archive.read_bytes()).hexdigest() != TINYGltf_ARCHIVE_SHA512:
        raise RuntimeError('Isolated tinygltf source archive changed')
    env.update(CMAKE_TOOLCHAIN_FILE=str(vcpkg / 'scripts/buildsystems/vcpkg.cmake'),
               VCPKG_MAX_CONCURRENCY=str(jobs), VCPKG_OVERLAY_PORTS=str(overlay.parent),
               CMAKE_GENERATOR='Ninja')
    # UIPC's developer-mode generator otherwise skips vcpkg on a retry even
    # when the preceding install failed partway through. Keep installed
    # packages/build outputs, but regenerate the manifest to resume safely.
    (tacex / 'tacex_uipc/build/vcpkg.json').unlink(missing_ok=True)
    run(pip + ['--no-deps', '--no-build-isolation', '-e', tacex / 'tacex_uipc'], env=env)
    curobo = home / 'curobo'
    if not curobo.exists():
        revision = subprocess.check_output(['git', '-C', str(root / 'UniVTAC/third_party/curobo'),
                                            'rev-parse', 'HEAD'], text=True).strip()
        if revision != CUROBO_COMMIT:
            raise RuntimeError(f'Unexpected cuRobo source revision: {revision}')
        ignore = shutil.ignore_patterns('.git', 'build', '*.so', '__pycache__', '*.egg-info')
        shutil.copytree(root / 'UniVTAC/third_party/curobo', curobo, ignore=ignore)
        setup = curobo / 'setup.py'
        setup.write_text(setup.read_text().replace('"--threads=8"', '"--threads=2"'))
    run(pip + ['--no-deps', '--no-build-isolation', '-e', curobo], env=env)
    run(pip + ['--no-deps', '-e', root / 'openpi/packages/openpi-client'], env=env)

    # The model retains its original Python 3.10 dependencies/source, but gets
    # an isolated CUDA-12.8 Torch build. Original site-packages stay read-only.
    model = home / 'model'
    model_python = model / 'bin/python'
    original_python = root / 'contact-belief-wam/.venv/bin/python'
    if not model_python.exists():
        run([original_python, '-m', 'venv', model], env=env)
    model_env = dict(env)
    model_env.pop('PIP_CONSTRAINT', None)
    model_env.pop('LD_LIBRARY_PATH', None)
    run([model_python, '-m', 'pip', 'install', 'torch==2.7.1', 'torchvision==0.22.1', 'torchaudio==2.7.1',
         '--index-url', 'https://download.pytorch.org/whl/cu128'], env=model_env)
    original_site = root / 'contact-belief-wam/.venv/lib/python3.10/site-packages'
    (model / 'lib/python3.10/site-packages/original_model_dependencies.pth').write_text(
        f'import site; site.addsitedir({str(original_site)!r})\n')
    code = "import json,torch; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'arch':torch._C._cuda_getArchFlags()}))"
    versions = {}
    for name, executable, child_env in [('sim', python, env), ('model', model_python, model_env)]:
        versions[name] = json.loads(subprocess.check_output([str(executable), '-c', code], env=child_env, text=True))
        if versions[name]['cuda'] != '12.8' or 'sm_120' not in versions[name]['arch']:
            raise RuntimeError(f'{name} Torch lacks native Blackwell support: {versions[name]}')
    bindings = list((prefix / 'lib/python3.11/site-packages/uipc').rglob('libuipc_backend_cuda.so'))
    bindings += list((curobo / 'src/curobo/curobolib').glob('*.so'))
    if len(bindings) != 6:
        raise RuntimeError(f'Expected one UIPC and five cuRobo CUDA binaries, found {len(bindings)}')
    scatter = list((prefix / 'lib/python3.11/site-packages/torch_scatter').glob('*_cuda.so'))
    # _version_cuda.so is a host-only version probe, with no CUDA kernels.
    scatter_kernels = [binary for binary in scatter if binary.name != '_version_cuda.so']
    if len(scatter_kernels) != 3:
        raise RuntimeError('Compiled torch_scatter CUDA modules missing')
    for binary in bindings + scatter_kernels:
        listing = subprocess.check_output([str(prefix / 'bin/cuobjdump'), '--list-elf', str(binary)], text=True)
        if any(arch not in listing for arch in NATIVE_ARCHITECTURES):
            raise RuntimeError(f'Ada/Blackwell machine code missing: {binary}')
    # Still integrity-check the version probe even though it has no device code.
    bindings += scatter
    graphics = prepare_graphics(root, home)
    write(home / 'ready.json', dict(profile=PROFILE, univtac_commit=UNIVTAC_COMMIT,
        training_simulator='4.5', evaluation_simulator='5.1', torch=versions,
        simulator_python=str(python), model_python=str(model_python), source=str(source),
        libraries={str(p): digest(p) for p in bindings},
        graphics_libraries={str(p): digest(p) for p in graphics.glob('*.so*') if p.is_file()},
        native_architectures=NATIVE_ARCHITECTURES,
        native_source_overrides={'tinygltf': {'version': '2.9.4', 'archive_sha512': TINYGltf_ARCHIVE_SHA512}},
        prepared_on=platform.node(), simulator_started=False))
    write(home / 'source.json', dict(profile=PROFILE, commit=UNIVTAC_COMMIT,
                                   files=source_inventory(source), missing_lfs=[]))


def repair_native(home, jobs):
    """Add Ada device code to the prepared extensions, retaining Blackwell code."""
    record = verify_ready(home)
    patch_source(Path(record['source']))
    prefix = home / 'sim'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        CUDA_HOME=str(prefix), CUDA_PATH=str(prefix), CUDACXX=str(prefix / 'bin/nvcc'),
        TORCH_CUDA_ARCH_LIST='8.9;12.0', CMAKE_CUDA_ARCHITECTURES='89-real;120-real',
        CMAKE_BUILD_PARALLEL_LEVEL=str(jobs), MAX_JOBS=str(jobs), NVCC_THREADS='2', FORCE_CUDA='1',
        CC=str(prefix / 'bin/x86_64-conda-linux-gnu-gcc'),
        CXX=str(prefix / 'bin/x86_64-conda-linux-gnu-c++'),
        CUDAHOSTCXX=str(prefix / 'bin/x86_64-conda-linux-gnu-c++'),
        SETUPTOOLS_SCM_PRETEND_VERSION='0.7.7',
        PATH=str(prefix / 'bin') + os.pathsep + os.environ.get('PATH', ''),
        LD_LIBRARY_PATH=str(prefix / 'lib'), CPATH=str(prefix / 'targets/x86_64-linux/include'),
        CMAKE_TOOLCHAIN_FILE=str(home / 'vcpkg/scripts/buildsystems/vcpkg.cmake'),
        VCPKG_MAX_CONCURRENCY=str(jobs), VCPKG_OVERLAY_PORTS=str(home / 'vcpkg-overlays'),
        CMAKE_GENERATOR='Ninja', PIP_INDEX_URL='https://pypi.org/simple',
        PIP_DISABLE_PIP_VERSION_CHECK='1')
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    # Same-version rebuilds overwrite the existing extension files. Avoid pip's
    # uninstall scan across the very large Isaac site-packages tree.
    pip = [prefix / 'bin/python', '-m', 'pip', 'install', '--ignore-installed',
           '--no-deps', '--no-build-isolation', '-v']
    groups = [
        ('uipc', None),
        ('curobo', ['-e', home / 'curobo']),
        ('torch_scatter', ['--force-reinstall', '--no-cache-dir', '--no-index',
                           home / 'downloads/torch_scatter-2.1.2.tar.gz']),
    ]
    for package, arguments in groups:
        binaries = [Path(p) for p in record['libraries'] if package in Path(p).parts]
        kernels = [p for p in binaries if p.name != '_version_cuda.so']
        def compiled():
            return bool(kernels) and all(all(arch in subprocess.check_output(
                [str(prefix / 'bin/cuobjdump'), '--list-elf', str(p)], text=True)
                for arch in NATIVE_ARCHITECTURES) for p in kernels)
        if not compiled():
            if package == 'uipc':
                project = home / 'UniVTAC/third_party/TacEx/source/tacex_uipc'
                build = project / 'build'
                run([prefix / 'bin/cmake', '-S', project / 'libuipc', '-B', build,
                     '-DCMAKE_CUDA_ARCHITECTURES=89-real;120-real'], env=env)
                run([prefix / 'bin/cmake', '--build', build, '--parallel', jobs], env=env)
                binary = build / 'Release/bin/libuipc_backend_cuda.so'
                listing = subprocess.check_output([str(prefix / 'bin/cuobjdump'), '--list-elf', str(binary)], text=True)
                if any(arch not in listing for arch in NATIVE_ARCHITECTURES):
                    raise RuntimeError('Rebuilt UIPC is missing Ada/Blackwell machine code')
                # Updating only the CUDA backend does not relink pyuipc, so its
                # POST_BUILD packaging hook is not run by incremental CMake.
                targets = binaries + [build / 'python/src/uipc/modules/Release/bin' / binary.name]
                for destination in targets:
                    temporary = destination.with_suffix(destination.suffix + '.partial')
                    shutil.copy2(binary, temporary)
                    temporary.replace(destination)
            else:
                if package == 'torch_scatter':
                    download(SCATTER_URL, home / 'downloads/torch_scatter-2.1.2.tar.gz', SCATTER_SHA256)
                run(pip + arguments, env=env)
        if not compiled():
            raise RuntimeError(f'{package} did not produce Ada and Blackwell machine code')
        record['libraries'].update({str(p): digest(p) for p in binaries})
        # Persist each completed group so a later interrupted build can resume.
        write(home / 'ready.json', record)
    record['native_architectures'] = NATIVE_ARCHITECTURES
    write(home / 'ready.json', record)
    write(home / 'source.json', dict(profile=PROFILE, commit=UNIVTAC_COMMIT,
          files=source_inventory(Path(record['source'])), missing_lfs=[]))
    verify_ready(home)


def verify_ready(home, *, require_graphics=True):
    """Read-only admission check: never install, fetch, compile, or start Kit."""
    ready = home / 'ready.json'
    if not ready.is_file():
        raise RuntimeError('Pro 6K environment is not prepared. Run prepare_pro6k.py on the development machine first; '
                           'this GPU job will not install dependencies.')
    record = json.loads(ready.read_text())
    if record.get('profile') != PROFILE or record.get('univtac_commit') != UNIVTAC_COMMIT:
        raise RuntimeError('Prepared environment has a different evaluation profile')
    if not record.get('libraries') or not record.get('graphics_libraries'):
        raise RuntimeError('Prepared environment is missing native/graphics library verification')
    required = {str(home / GRAPHICS_DIR / 'lib' / name) for name in GRAPHICS_SONAMES}
    if require_graphics and not required.issubset(record['graphics_libraries']):
        raise RuntimeError('Prepared environment is missing the matching GLVND stack; run '
                           'prepare_pro6k.py --repair-graphics on the development machine')
    for group in ('libraries', 'graphics_libraries'):
        for name, expected in record[group].items():
            path = Path(name)
            if not path.is_file() or digest(path) != expected:
                raise RuntimeError(f'Prepared library is missing or changed: {path}')
    for key in ('simulator_python', 'model_python'):
        if not Path(record[key]).is_file():
            raise RuntimeError(f'Prepared Python is missing: {record[key]}')
    source_record = json.loads((home / 'source.json').read_text())
    source = Path(record['source'])
    if source_record['commit'] != UNIVTAC_COMMIT or source_record.get('missing_lfs'):
        raise RuntimeError('Prepared simulator source is incomplete')
    for name, expected in source_record['files'].items():
        path = source / name
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f'Prepared simulator source changed: {path}')
    print(f'Prepared Pro 6K environment verified (no installation): {home}', flush=True)
    return record


def require_contact_reset(record):
    patch = record.get('contact_reset_patch', {})
    if (patch.get('id') != CONTACT_RESET_PATCH
            or patch.get('source_sha256') != CONTACT_PATCHED_SHA256
            or not patch.get('binary_sha256')
            or record['libraries'].get(patch.get('binary')) != patch['binary_sha256']):
        raise RuntimeError('Prepare the contact-reset repair on the development machine first: '
                           'prepare_pro6k.py --repair-contact-reset')
    source = Path(record['source']) / CONTACT_SOURCE
    if not source.is_file() or digest(source) != patch['source_sha256']:
        raise RuntimeError('Contact-reset source no longer matches the prepared binary')


def repair_contact_reset(home, jobs):
    """Rebuild the contact-cache fix without reinstalling Isaac or other packages."""
    record = verify_ready(home)
    if record.get('contact_reset_patch', {}).get('id') == CONTACT_RESET_PATCH:
        require_contact_reset(record)
        return
    source = Path(record['source']) / CONTACT_SOURCE
    current = digest(source)
    if current == CONTACT_ORIGINAL_SHA256:
        old = '''            // 1. Record Friction Candidates at the beginning of the frame
            record_friction_candidates();
            m_global_vertex_manager->update_attributes();
            m_global_vertex_manager->record_prev_positions();

            // 2. Adaptive Parameter Calculation
            detect_dcd_candidates();'''
        new = '''            // Restore/state-accessor writes can teleport geometry between frames.
            // Refresh contact pairs at the current positions before freezing the
            // lagged friction set; pairs cached before a reset are not valid here.
            m_global_vertex_manager->update_attributes();
            m_global_vertex_manager->record_prev_positions();
            detect_dcd_candidates();
            record_friction_candidates();

            // 2. Adaptive Parameter Calculation'''
        text = source.read_text()
        if text.count(old) != 1:
            raise RuntimeError('Unknown UIPC contact update sequence')
        source.write_text(text.replace(old, new))
    elif current != CONTACT_PATCHED_SHA256:
        raise RuntimeError('Unknown UIPC source revision; refusing to apply the contact-reset patch')
    if digest(source) != CONTACT_PATCHED_SHA256:
        raise RuntimeError('Unexpected patched UIPC source checksum')
    prefix = home / 'sim'
    project = home / 'UniVTAC/third_party/TacEx/source/tacex_uipc'
    build = project / 'build'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_HOME=str(prefix),
        CUDA_PATH=str(prefix), CUDACXX=str(prefix / 'bin/nvcc'),
        PATH=str(prefix / 'bin') + os.pathsep + os.environ.get('PATH', ''),
        LD_LIBRARY_PATH=str(prefix / 'lib'), CPATH=str(prefix / 'targets/x86_64-linux/include'))
    for key in ('PYTHONHOME', 'PYTHONPATH'):
        env.pop(key, None)
    run([prefix / 'bin/cmake', '-S', project / 'libuipc', '-B', build,
         '-DCMAKE_CUDA_ARCHITECTURES=89-real;120-real'], env=env)
    run([prefix / 'bin/cmake', '--build', build, '--target', 'cuda', '--parallel', jobs], env=env)
    binary = build / 'Release/bin/libuipc_backend_cuda.so'
    listing = subprocess.check_output([str(prefix / 'bin/cuobjdump'), '--list-elf', str(binary)], text=True)
    if any(arch not in listing for arch in NATIVE_ARCHITECTURES):
        raise RuntimeError('Contact-reset backend lacks Ada or Blackwell machine code')
    installed, = [Path(p) for p in record['libraries'] if p.endswith('/libuipc_backend_cuda.so')]
    previous_sha = digest(installed)
    backup = home / 'native-backups' / CONTACT_RESET_PATCH / (previous_sha + '.so')
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(installed, backup)
    if digest(backup) != previous_sha:
        raise RuntimeError('The original UIPC backend backup is incomplete')
    for destination in (installed, build / 'python/src/uipc/modules/Release/bin' / binary.name):
        temporary = destination.with_suffix('.so.partial')
        shutil.copy2(binary, temporary)
        temporary.replace(destination)
    record['libraries'][str(installed)] = digest(installed)
    record['contact_reset_patch'] = dict(id=CONTACT_RESET_PATCH, source_sha256=digest(source),
        binary=str(installed), binary_sha256=digest(installed), prepared_on=platform.node(),
        original_binary_sha256=previous_sha, rollback_binary=str(backup))
    # Track this native source explicitly; the general inventory covers Python
    # and build configuration rather than the whole vendored C++ tree.
    source_record = json.loads((home / 'source.json').read_text())
    source_record['files'][CONTACT_SOURCE] = digest(source)
    write(home / 'source.json', source_record)
    write(home / 'ready.json', record)
    require_contact_reset(verify_ready(home))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--sources-only', action='store_true')
    parser.add_argument('--check-only', action='store_true', help='Verify the prepared environment without changing it')
    parser.add_argument('--repair-graphics', action='store_true',
                        help='Prepare matching GLVND on the development machine without rebuilding Isaac/CUDA')
    parser.add_argument('--repair-native', action='store_true',
                        help='Build Ada and Blackwell extensions on the development machine without reinstalling Isaac')
    parser.add_argument('--repair-contact-reset', action='store_true',
                        help='Repair stale UIPC friction contacts after reset, on the development machine')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if sum((args.repair_graphics, args.repair_native, args.repair_contact_reset,
            args.check_only, args.sources_only)) > 1:
        parser.error('Choose only one preparation/check mode')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    root = args.root.resolve()
    home = root / '.cache' / ENV_NAME
    if args.check_only:
        verify_ready(home)
        return
    if not args.sources_only and os.environ.get('AIHC_JOB_NAME'):
        raise RuntimeError('Install this environment on the development machine, not inside an AIHC GPU job')
    home.mkdir(parents=True, exist_ok=True)
    with (home / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.repair_contact_reset:
            repair_contact_reset(home, args.jobs)
            return
        if args.repair_native:
            repair_native(home, args.jobs)
            return
        if args.repair_graphics:
            record = verify_ready(home, require_graphics=False)
            graphics = prepare_graphics(root, home)
            record['graphics_libraries'] = {str(p): digest(p) for p in graphics.glob('*.so*') if p.is_file()}
            record['graphics_repair'] = dict(build=GLVND_BUILD,
                                            packages={k: v[0] for k, v in GLVND_PACKAGES.items()},
                                            prepared_on=platform.node(), gpu_initialized=False)
            write(home / 'ready.json', record)
            verify_ready(home)
            return
        if not args.sources_only:
            if platform.machine() != 'x86_64' or tuple(map(int, platform.libc_ver()[1].split('.')[:2])) < (2, 35):
                raise RuntimeError('Isaac Sim 5.1 needs x86_64 Linux with glibc >= 2.35 (e.g. Ubuntu 22.04)')
        source, missing = prepare_source(root, home, fetch=not args.sources_only)
        if args.sources_only:
            return
        ready = home / 'ready.json'
        if ready.exists():
            verify_ready(home)
        else:
            if missing:
                raise RuntimeError(f'Missing LFS assets: {missing}')
            install(root, home, source, args.jobs)


if __name__ == '__main__':
    main()
