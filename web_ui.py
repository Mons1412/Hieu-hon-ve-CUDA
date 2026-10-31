from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from api import app
from pdf_rag import PDF_DIR

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def homepage():
    return """
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CUDA Knowledge Assistant</title>

<style>
*{box-sizing:border-box}
body{margin:0;background:#0d1423;color:#edf3ff;font:15px/1.6 system-ui}
#layout{display:grid;grid-template-columns:250px 1fr;min-height:100vh}
aside{background:#152136;padding:22px;border-right:1px solid #33445e}
main{display:flex;flex-direction:column;height:100vh}
header{padding:18px 28px;border-bottom:1px solid #33445e}
#feed{flex:1;overflow:auto;padding:25px max(25px,8%)}
article{background:#19283e;margin:12px 0;padding:15px;border-radius:12px;white-space:pre-wrap;overflow-wrap:anywhere}
.user{background:#203c56}
small{color:#b2c2d6}
a{color:#81ceff}
.source{margin-top:9px;padding:10px;border:1px solid #455a76;border-radius:8px}
form{display:flex;gap:12px;padding:20px;background:#152136}
textarea{flex:1;min-height:66px;background:#0c1525;color:white;border:1px solid #4a607e;border-radius:8px;padding:12px;font:inherit}
button{background:#2678af;color:white;border:0;border-radius:8px;padding:12px;cursor:pointer}
button:disabled{opacity:.45}
h3{color:#8fcfff}
@media(max-width:700px){
#layout{display:block}
aside{padding:12px}
main{height:80vh}
#feed{padding:10px}
}
</style>
</head>

<body>
<div id="layout">
<aside>
<h2>⌘ CUDA Assistant</h2>
<small>Local LLM + RAG v0.5</small>

<h3>Trạng thái</h3>
<div id="status">Đang kiểm tra...</div>

<h3>Model / Chunks</h3>
<div id="model">—</div>

<h3>Tài liệu</h3>
<div id="docs">—</div>

<p><button id="reset">Chat mới</button></p>
<a href="/docs">Swagger API ↗</a>
</aside>

<main>
<header><b>CUDA Knowledge Assistant</b> · NVIDIA PDF Research</header>

<section id="feed">
<h2>Hỏi về CUDA, Shared Memory, Matrix Multiplication...</h2>
<small>Kết quả có nguồn PDF; AI vẫn có thể sai.</small>
</section>

<form id="form">
<textarea id="q" placeholder="Nhập câu hỏi (Enter để gửi)" maxlength="3000" required></textarea>
<button id="send" disabled>Gửi ➜</button>
</form>
</main>
</div>

<script>
const $=id=>document.getElementById(id);
let ready=false,busy=false;

function add(tag,text,cls){
 const n=document.createElement(tag);
 n.textContent=text||'';
 if(cls)n.className=cls;
 return n;
}

async function json(url,opt){
 const r=await fetch(url,opt);
 const d=await r.json();
 if(!r.ok)throw Error(d.detail||r.status);
 return d;
}

async function setup(){
 try{
  const s=await json('/health/ready');
  ready=true;
  $('status').textContent='🟢 Ready';
  $('model').textContent=s.llm+' / '+s.chunks+' chunks';

  const d=await json('/api/v1/documents');
  $('docs').textContent=d.documents.map(
   x=>x.source+' ('+x.chunks+' chunks)'
  ).join(', ');
 }catch(e){
  ready=false;
  $('status').textContent='🔴 '+e.message;
 }
 $('send').disabled=!ready;
}

function message(label,text,cls){
 const n=add('article','',cls);
 n.append(add('b',label));
 n.append(add('div',text));
 $('feed').append(n);
 $('feed').scrollTop=$('feed').scrollHeight;
 return n;
}

async function send(q){
 if(!ready||busy)return;

 busy=true;
 $('send').disabled=true;

 message('Bạn',q,'user');
 const out=message('Assistant','Đang tìm nguồn...');

 try{
  const d=await json('/api/v1/chat',{
   method:'POST',
   headers:{'Content-Type':'application/json'},
   body:JSON.stringify({question:q,top_k:4})
  });

  out.children[1].textContent=
   d.status==='ANSWER'?d.answer:
   d.status==='ABSTAIN'?'Không đủ dữ liệu.':
   'Câu trả lời chưa đạt kiểm tra định dạng.';

  out.append(add('small',
   'Trạng thái: '+d.status+
   ' | Chưa kiểm chứng sự thật'
  ));

  for(const s of d.sources||[]){
   const c=add('div','','source');
   const a=add('a',
    '['+s.rank+'] '+s.source+
    ' — PDF trang '+s.page+' ↗'
   );

   a.href='/api/v1/pdf?source='+
    encodeURIComponent(s.source)+'#page='+s.page;
   a.target='_blank';
   a.rel='noopener noreferrer';
   c.append(a);

   const det=document.createElement('details');
   det.append(add('summary','Xem đoạn nguồn'));
   det.append(add('p',s.text||s.preview));
   c.append(det);
   out.append(c);
  }
 }catch(e){
  out.children[1].textContent='Lỗi: '+e.message;
 }finally{
  busy=false;
  $('send').disabled=!ready;
 }
}

$('form').onsubmit=e=>{
 e.preventDefault();
 const q=$('q').value.trim();
 if(q){
  $('q').value='';
  send(q);
 }
};

$('q').onkeydown=e=>{
 if(e.key==='Enter'&&!e.shiftKey){
  e.preventDefault();
  $('form').requestSubmit();
 }
};

$('reset').onclick=()=>{
 if(!busy)$('feed').replaceChildren(
  add('h2','Cuộc trò chuyện mới')
 );
};

setup();
</script>
</body>
</html>
"""

@app.get("/api/v1/pdf", include_in_schema=False)
def pdf(source: str):
    root = Path(PDF_DIR).resolve()
    target = (root / source).resolve()

    if (
        not target.is_relative_to(root)
        or target.suffix.lower() != ".pdf"
        or not target.is_file()
    ):
        raise HTTPException(404, "PDF not found")

    return FileResponse(target, media_type="application/pdf")
