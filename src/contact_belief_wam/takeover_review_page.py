"""Local, offline frame-by-frame review page; no server or third-party assets."""

PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HDMI · 两条示范逐帧核对</title>
<style>
:root{color-scheme:dark;--bg:#10151e;--panel:#192230;--line:#344355;--text:#edf3fb;--muted:#afbdce;--good:#70dfb0;--warn:#ffcc80}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.65 system-ui,sans-serif}
main{max-width:1280px;margin:28px auto;padding:0 22px}h1{font-size:28px;line-height:1.3;margin:10px 0}h2{font-size:18px;margin:0 0 12px}
p{margin:8px 0 14px}.muted{color:var(--muted)}a{color:#a3cbff}.pill{display:inline-block;border:1px solid var(--line);padding:3px 10px;border-radius:20px;margin-right:8px}.good{color:var(--good)}.warn{color:var(--warn)}
section{padding:20px;margin:18px 0;background:var(--panel);border:1px solid var(--line);border-radius:12px}
button,select,input[type=number]{background:#243348;border:1px solid #4b607b;color:var(--text);padding:8px 13px;border-radius:7px;font:inherit}button{cursor:pointer}button:hover{background:#334960}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid #9bcaff}button.active{border-color:var(--good)}
.controls{position:sticky;top:0;z-index:3;background:#192230f5;backdrop-filter:blur(8px)}.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap}input[type=range]{width:100%;accent-color:var(--good);margin:16px 0}input[type=number]{width:100px}.layout{display:grid;grid-template-columns:minmax(300px,448px) minmax(360px,1fr);gap:22px}.frame{width:100%;aspect-ratio:1;background:#0b1018;border-radius:5px}.viewlabels{display:grid;grid-template-columns:1fr 1fr;text-align:center;color:var(--muted);font-size:13px}canvas{display:block;max-width:100%;height:auto}.bar{display:grid;grid-template-columns:repeat(50,1fr);gap:2px;margin:12px 0}.bar i{height:15px;background:#3e4b5c}.bar i.yes{background:var(--good)}
pre{font:12px/1.55 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#101722;padding:12px;border-radius:6px}table{border-collapse:collapse;width:100%;font:13px/1.6 ui-monospace,monospace}td,th{text-align:right;padding:5px 7px;border-bottom:1px solid var(--line)}td:first-child,th:first-child{text-align:left}.scroll{overflow-x:auto}.timeline{display:flex;gap:6px;flex-wrap:wrap}.timeline button{padding:4px 9px;font-size:13px}details{margin:14px 0}summary{cursor:pointer}.callout{border-left:3px solid var(--warn);padding-left:12px}.loading{opacity:.45}.historylabels{display:grid;grid-template-columns:repeat(4,1fr);text-align:center;color:var(--muted)}.footer{font-size:13px;color:var(--muted);margin:24px 0}
@media(max-width:800px){main{padding:0 12px}.layout{grid-template-columns:1fr}section{padding:14px}h1{font-size:23px}.controls{position:static}}
</style></head><body><main>
<div class="muted">CONTACT BELIEF / HDMI / DATA REVIEW</div><h1>一条恢复，一条正常成功</h1>
<p>逐帧核对真实观测、已执行历史和下一步命令。候选监督独立保存；原始轨迹及原训练清单保持不变。</p>
<div><span class="pill good">数据接口检查通过</span><span class="pill warn">仅审核 · 未启动训练</span><span class="pill">未新增采集</span></div>
<section class="controls"><div class="row"><label for="episode">示范</label><select id="episode"></select><button id="prev">← 上一帧</button><button id="play">播放</button><button id="next">下一帧 →</button><label for="number">观测 t</label><input id="number" type="number" min="0" value="40"><span id="counter"></span></div>
<input id="slider" type="range" min="0" value="40" aria-label="观测帧"><div class="timeline" id="jumps"></div><div class="muted">← / → 单步；空格播放或暂停。所有区间均为 [开始, 结束)，观测 oₜ 对应命令 aₜ 和后续观测 oₜ₊₁。</div></section>
<section><div class="layout"><div><h2>当前双视角与触觉</h2><div class="viewlabels"><span>HEAD · 模型裁放方式</span><span>WRIST · 模型裁放方式</span></div><img class="frame" id="frame" alt="当前双视角和左右触觉"><div class="viewlabels"><span>左触觉 · 当前帧</span><span>右触觉 · 当前帧</span></div><p class="muted">图像为压缩预览；训练从原始 HDF5 / AnyTouch 缓存读取。精确输入张量另存于 samples/。</p></div>
<div><h2 id="stage"></h2><p id="label"></p><div class="bar" id="mask"></div><p class="muted">未来 50 个动作位置：绿色为候选 BC 标签；灰色在生成带噪策略输入前已替换，且不计策略 loss。</p><pre id="context"></pre><div class="scroll"><table><thead><tr><th>维度</th><th>当前 qₜ</th><th>下发 aₜ</th><th>后续 qₜ₊₁</th></tr></thead><tbody id="joints"></tbody></table></div><p class="muted">前七维为关节角；最后一维为夹爪。动作标签使用实际下发命令，不用后续实测状态替代。</p></div></div></section>
<section><h2>模型可用的四帧触觉窗口</h2><div id="windowlabels" class="historylabels"></div><canvas id="windows" width="896" height="448" aria-label="左触觉四帧在上、右触觉四帧在下"></canvas><p class="muted">上排左触觉，下排右触觉。窗口严格使用 t−6、t−4、t−2、t；仅轨迹开始处补首帧，接管和标签边界不重置。AnyTouch2 latent 来自该连续窗口。</p><p id="touchmetrics"></p></section>
<section><h2>连续历史与标签依据</h2><details open><summary>最近最多 10 个已执行命令</summary><pre id="history"></pre></details><details><summary>本条候选监督区间</summary><pre id="ranges"></pre></details><details><summary>仿真真值，仅用于 Teacher 来源说明与审核，不进入学生输入</summary><pre id="truth"></pre></details><p id="timing"></p></section>
<section><h2>验收记录</h2><p id="audit"></p><p class="callout">恢复 Teacher 使用真实横向误差与真实插入进展。本页验证数据接口、时序和标签隔离，不证明学生已经学会触觉恢复。接管时像素一致包含恢复图像缓存的效果，不能证明后续渲染逐像素一致；跨分支逐像素一致不是模仿学习的必要门槛。</p><p id="pixel-diagnosis" hidden></p><div class="row"><a href="validation.json">完整检查结果</a><a href="review_manifest.jsonl">隔离审核清单</a><a href="contracts.json">样本契约</a><a href="loader_checks.json">每个锚点的加载检查</a><a id="csv">逐帧 CSV</a></div></section>
<p class="footer">正常成功完整保留有效监督；恢复前缀保留为历史和 WM 转移，暂不列为策略正样本。Teacher 的每次新决策单独限定标签终点，禁止较早输入借用后来反馈产生的标签。</p>
</main><script src="review_data.js"></script><script>
'use strict';
const $=id=>document.getElementById(id), data=window.REVIEW;
let e=0,t=40,timer=null,renderId=0;
const names={natural:'WAVE-0 自然执行',oracle_unload:'卸载接触',oracle_align_xy:'真值辅助横向纠偏',oracle_insert:'小步插入',terminal_success:'成功终止'};
data.episodes.forEach((ep,i)=>{const o=document.createElement('option');o.value=i;o.textContent=ep.seed+' · '+ep.title;$('episode').append(o)});
function current(){return data.episodes[e]}
function path(step){return 'frames/'+current().seed+'/'+String(step).padStart(4,'0')+'.jpg'}
function fmt(x){return x===null||x===undefined?'—':Number(x).toFixed(6)}
function pause(){clearInterval(timer);timer=null;$('play').textContent='播放'}
function setFrame(value){t=Math.max(0,Math.min(current().actions,Number(value)||0));render()}
function image(path){return new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error('无法读取 '+path));im.src=path})}
async function render(){
 const id=++renderId,ep=current(),f=ep.frames[t];
 $('slider').max=ep.actions;$('slider').value=t;$('number').max=ep.actions;$('number').value=t;$('counter').textContent='/ '+ep.actions;
 $('frame').src=path(t);$('stage').textContent='t = '+t+' · '+(names[f.phase]||f.phase);
 $('label').textContent=f.policy_valid_prefix?'候选监督 a['+t+':'+f.label_stop+']，共 '+f.policy_valid_prefix+' 步。':'此锚点无策略正标签；保留可用的历史与真实动力学转移。';
 $('mask').replaceChildren(...Array.from({length:50},(_,i)=>{const x=document.createElement('i');if(i<f.policy_valid_prefix)x.className='yes';x.title='动作 '+(t+i)+(i<f.policy_valid_prefix?' 有效':' 排除 / 补齐');return x}));
 const begin=Math.max(0,t-10),warm=[];for(let k=Math.min(4,Math.floor(t/10));k>=0;k--)warm.push(t-k*10);
 $('context').textContent='该动作的 Teacher / WAVE-0 决策时刻：'+(f.decision??'终止')+'\nBelief 连续观测：['+warm.join(', ')+']\n已执行命令：a['+begin+':'+t+']\n触觉窗口：['+f.touch_window.join(', ')+']\n本动作物理步数：'+(f.substeps??'无下一动作')+'\nWM 的 t+10 转移：'+(t+10<=ep.actions?'有效，保留':'超出轨迹，不计目标');
 $('joints').replaceChildren(...f.q.map((v,i)=>{const tr=document.createElement('tr');[i===7?'夹爪':'q'+(i+1),fmt(v),fmt(f.command?.[i]),fmt(f.next_q?.[i])].forEach(s=>{const td=document.createElement('td');td.textContent=s;tr.append(td)});return tr}));
 $('windowlabels').replaceChildren(...f.touch_window.map(s=>{const el=document.createElement('span');el.textContent='观测 '+s;return el}));
 $('touchmetrics').textContent='相邻原始触觉平均变化（/255）：左 '+fmt(f.raw_touch_delta[0])+'，右 '+fmt(f.raw_touch_delta[1])+'；AnyTouch2 相对初始参考的平均 cosine 变化：'+fmt(f.latent_delta)+'。';
 $('history').textContent=ep.frames.slice(begin,t).map(x=>'a['+x.t+'] '+x.command.map(fmt).join('  ')).join('\n')||'尚无已执行命令';
 $('ranges').textContent=JSON.stringify(ep.ranges,null,2);$('truth').textContent=JSON.stringify({object_minus_slot_world_mm:f.audit_only_object_slot_mm,privileged_teacher:ep.takeover!==null,student_input_contains_truth:false},null,2);
 $('timing').textContent=ep.takeover!==null?'本条标注：接触接近证据 t='+ep.timing.t_contact+' → 触觉新证据 t='+ep.timing.t_tactile+' → 接管决策 / 首次恢复 t='+ep.takeover+'。接触时刻来自表面距离代理，并非接触力测量。':'本条正常成功未经恢复接管；接触、短暂停滞等片段继续保留为正常控制示范。';
 $('csv').href='frames-'+ep.seed+'.csv';
 const canvas=$('windows');canvas.classList.add('loading');
 try{const ims=await Promise.all(f.touch_window.map(s=>image(path(s))));if(id!==renderId)return;const ctx=canvas.getContext('2d');ctx.clearRect(0,0,896,448);ims.forEach((im,i)=>{ctx.drawImage(im,0,224,224,224,i*224,0,224,224);ctx.drawImage(im,224,224,224,224,i*224,224,224,224)});canvas.classList.remove('loading')}catch(err){if(id===renderId)$('touchmetrics').textContent=err.message}
}
function jumps(){const ep=current();const points=new Set([0,ep.actions]);if(ep.takeover!==null)[30,37,38,39,40,59,60,109,110,197].forEach(t=>points.add(t));else [40,100,200,300,400,500,546].forEach(t=>points.add(t));$('jumps').replaceChildren(...[...points].sort((a,b)=>a-b).map(v=>{const b=document.createElement('button');b.textContent=v;b.onclick=()=>{pause();setFrame(v)};return b}))}
$('episode').onchange=()=>{pause();e=Number($('episode').value);jumps();setFrame(current().takeover??0)};
$('prev').onclick=()=>{pause();setFrame(t-1)};$('next').onclick=()=>{pause();setFrame(t+1)};
$('slider').oninput=()=>{pause();setFrame($('slider').value)};$('number').onchange=()=>{pause();setFrame($('number').value)};
$('play').onclick=()=>{if(timer){pause();return}if(t===current().actions)setFrame(0);$('play').textContent='暂停';timer=setInterval(()=>{if(t>=current().actions){pause();return}setFrame(t+1)},160)};
document.addEventListener('keydown',ev=>{if(['INPUT','SELECT','TEXTAREA'].includes(ev.target.tagName))return;if(ev.code==='ArrowLeft'||ev.code==='ArrowRight'){ev.preventDefault();pause();setFrame(t+(ev.code==='ArrowLeft'?-1:1))}else if(ev.code==='Space'){ev.preventDefault();$('play').click()}});
$('audit').textContent='已检查 '+data.validation.loader.anchors_checked+' 个动作锚点、全部标签边界与连续历史；'+data.validation.loader.policy_supervised_anchors+' 个锚点具有候选策略监督。8 个关键观测重新执行了冻结 AnyTouch2 编码。正常与恢复原始文件哈希核对通过。';
if(data.pixel_diagnosis){const p=$('pixel-diagnosis'),a=document.createElement('a');a.href='pixel_diagnosis/index.html';a.textContent='像素差异诊断：这条恢复可用于真值 Teacher 模仿学习';p.hidden=false;p.append(a)}
jumps();render();
</script></body></html>'''
