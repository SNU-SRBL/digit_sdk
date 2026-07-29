"use strict";
const $ = (id) => document.getElementById(id);
const view = $("view");
const ctx = view.getContext("2d");
const source = document.createElement("canvas");
const sourceCtx = source.getContext("2d", {willReadFrequently:true});
const difference = document.createElement("canvas");
const differenceCtx = difference.getContext("2d", {willReadFrequently:true});
const state = {samples:[], index:0, center:{x:0,y:0}, radius:20, dragging:false, dirty:false, undo:[], redo:[], hasBackground:false};

function status(message, kind="") { $("status").textContent=message; $("status").className=kind; }
function queueStatus(value) { $("queue").textContent=`${value.completed}/${value.total} labeled · ${value.rejected} rejected · ${value.remaining} remaining`; }
function loadImage(url) { return new Promise((resolve,reject)=>{ const image=new Image(); image.onload=()=>resolve(image); image.onerror=()=>reject(new Error(`Could not load ${url}`)); image.src=`${url}?t=${Date.now()}`; }); }
function applyZoom() { const scale=Number($("zoom").value)/100; view.style.width=`${view.width*scale}px`; view.style.height=`${view.height*scale}px`; $("zoom-value").value=`${$("zoom").value}%`; }
function point(event) { const box=view.getBoundingClientRect(); return {x:(event.clientX-box.left)*view.width/box.width,y:(event.clientY-box.top)*view.height/box.height}; }
function current() { return state.samples[state.index]; }
function editable() { return !current().rejected; }
function maxRadius() { return current().ball_diameter_mm*current().ppmm/2-0.01; }
function peakDepth() { const R=current().ball_diameter_mm/2; const a=state.radius/current().ppmm; return R-Math.sqrt(R*R-a*a); }
function updateGeometry() {
  state.radius=Math.max(0.25,Math.min(maxRadius(),state.radius));
  state.center.x=Math.max(0,Math.min(view.width-1,state.center.x)); state.center.y=Math.max(0,Math.min(view.height-1,state.center.y));
  $("radius").max=maxRadius(); $("radius").value=state.radius; $("radius-value").value=`${state.radius.toFixed(2)} px`;
  $("diameter").textContent=`${current().ball_diameter_mm.toFixed(3)} mm`; $("depth").textContent=`${peakDepth().toFixed(4)} mm`; $("center").textContent=`${state.center.x.toFixed(1)}, ${state.center.y.toFixed(1)}`;
}
function render() {
  if (!view.width) return; ctx.clearRect(0,0,view.width,view.height); ctx.drawImage($("difference").checked&&state.hasBackground?difference:source,0,0);
  ctx.save(); ctx.strokeStyle="#ffbf3f"; ctx.lineWidth=1.5; ctx.setLineDash([5,4]); ctx.beginPath(); ctx.arc(state.center.x,state.center.y,state.radius,0,Math.PI*2); ctx.stroke(); ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(state.center.x-5,state.center.y); ctx.lineTo(state.center.x+5,state.center.y); ctx.moveTo(state.center.x,state.center.y-5); ctx.lineTo(state.center.x,state.center.y+5); ctx.stroke(); ctx.restore();
  updateGeometry();
}
function snapshot() { return {center:{...state.center},radius:state.radius}; }
function pushUndo() { state.undo.push(snapshot()); if(state.undo.length>40)state.undo.shift(); state.redo=[]; historyButtons(); }
function historyButtons() { $("undo").disabled=!state.undo.length; $("redo").disabled=!state.redo.length; }
function restore(value) { state.center={...value.center}; state.radius=value.radius; markDirty(); render(); }
function markDirty() { state.dirty=true; status("Unsaved changes"); }
function updateActions() { const rejected=current().rejected; $("reject").textContent=rejected?"Undo rejection":"Reject sample"; $("reject").classList.toggle("danger",!rejected); $("save").disabled=rejected; }
function makeDifference(background) { difference.width=view.width; difference.height=view.height; differenceCtx.drawImage(background,0,0); const a=sourceCtx.getImageData(0,0,view.width,view.height),b=differenceCtx.getImageData(0,0,view.width,view.height),o=differenceCtx.createImageData(view.width,view.height); for(let i=0;i<o.data.length;i+=4){o.data[i]=Math.min(255,Math.abs(a.data[i]-b.data[i])*3);o.data[i+1]=Math.min(255,Math.abs(a.data[i+1]-b.data[i+1])*3);o.data[i+2]=Math.min(255,Math.abs(a.data[i+2]-b.data[i+2])*3);o.data[i+3]=255;} differenceCtx.putImageData(o,0,0); }
async function loadSample(index) {
  state.index=index; const sample=current(); status("Loading image…"); const image=await loadImage(sample.image_url); view.width=source.width=image.naturalWidth; view.height=source.height=image.naturalHeight; sourceCtx.drawImage(image,0,0); applyZoom();
  state.center=sample.center_px?{x:sample.center_px[0],y:sample.center_px[1]}:{x:view.width/2,y:view.height/2}; state.radius=sample.radius_px||Math.min(40,maxRadius()/2); state.hasBackground=false; $("difference").disabled=!sample.background_url;
  if(sample.background_url){try{makeDifference(await loadImage(sample.background_url));state.hasBackground=true;$("difference").disabled=false;}catch(error){console.warn(error);}}
  state.undo=[];state.redo=[];state.dirty=false;historyButtons(); $("sample-id").textContent=sample.sample_id; const sampleState=sample.rejected?`rejected by ${sample.rejected_by}`:(sample.saved?`saved r${sample.annotation_revision}`:"draft"); $("sample-meta").textContent=`${index+1}/${state.samples.length} · ${sample.session_id} · ${sampleState}`; $("previous").disabled=index===0; $("next").disabled=index===state.samples.length-1; status(sample.rejected?"Rejected from dataset":(sample.saved?`Saved revision ${sample.annotation_revision}`:"Not labeled"),sample.rejected?"rejected":(sample.saved?"saved":"")); updateActions(); render();
}
async function save() {
  if(current().rejected){status("Undo rejection before saving","error");return false;}
  const annotator=$("annotator").value.trim(); if(!annotator){status("Enter an annotator name","error");return false;} localStorage.setItem("digit-annotator",annotator); status("Saving…");
  try { const response=await fetch(current().image_url.replace(/\/image$/, "/annotation"),{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({center_px:[state.center.x,state.center.y],radius_px:state.radius,annotator})}); const payload=await response.json(); if(!response.ok)throw new Error(payload.error||"Save failed"); Object.assign(current(),payload); state.dirty=false; status(`Saved revision ${payload.annotation_revision}`,"saved"); await refresh(); return true; } catch(error){status(error.message,"error");return false;}
}
async function toggleRejection(){const annotator=$("annotator").value.trim();if(!annotator){status("Enter an annotator name","error");return;}localStorage.setItem("digit-annotator",annotator);const rejected=!current().rejected;status(rejected?"Rejecting…":"Restoring…");try{const response=await fetch(current().image_url.replace(/\/image$/,"/rejection"),{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({rejected,annotator})});const payload=await response.json();if(!response.ok)throw new Error(payload.error||"Rejection update failed");Object.assign(current(),payload);if(!payload.rejected){current().rejected_by="";current().rejection_reason="";}state.dirty=false;await refresh();await loadSample(state.index);}catch(error){status(error.message,"error");}}
async function refresh(){const response=await fetch("/api/status",{cache:"no-store"});const value=await response.json();queueStatus(value);}
async function navigate(offset){if(state.dirty&&!(await save()))return;const next=Math.max(0,Math.min(state.samples.length-1,state.index+offset));if(next!==state.index)await loadSample(next);}
view.addEventListener("pointerdown",event=>{if(!editable())return;pushUndo();state.dragging=true;state.center=point(event);view.setPointerCapture(event.pointerId);markDirty();render();});
view.addEventListener("pointermove",event=>{if(!state.dragging)return;state.center=point(event);markDirty();render();});
view.addEventListener("pointerup",event=>{state.dragging=false;if(view.hasPointerCapture(event.pointerId))view.releasePointerCapture(event.pointerId);});
view.addEventListener("wheel",event=>{event.preventDefault();if(!editable())return;pushUndo();state.radius+=event.deltaY<0?0.5:-0.5;markDirty();render();},{passive:false});
$("radius").addEventListener("input",()=>{if(!editable())return;pushUndo();state.radius=Number($("radius").value);markDirty();render();});
$("zoom").addEventListener("input",applyZoom); $("difference").addEventListener("change",render); $("reject").addEventListener("click",toggleRejection); $("save").addEventListener("click",save); $("previous").addEventListener("click",()=>navigate(-1)); $("next").addEventListener("click",()=>navigate(1));
$("undo").addEventListener("click",()=>{if(!state.undo.length)return;state.redo.push(snapshot());restore(state.undo.pop());historyButtons();}); $("redo").addEventListener("click",()=>{if(!state.redo.length)return;state.undo.push(snapshot());restore(state.redo.pop());historyButtons();});
document.addEventListener("keydown",event=>{if(event.target instanceof HTMLInputElement&&event.target.type!=="range")return;if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="z"){event.preventDefault();(event.shiftKey?$("redo"):$("undo")).click();return;}if(event.key==="ArrowLeft"&&event.shiftKey){navigate(-1);return;}if(event.key==="ArrowRight"&&event.shiftKey){navigate(1);return;}if(!editable())return;const move={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[event.key];if(move){pushUndo();state.center.x+=move[0];state.center.y+=move[1];markDirty();render();}else if(event.key==="+"||event.key==="="){pushUndo();state.radius+=0.5;markDirty();render();}else if(event.key==="-"){pushUndo();state.radius-=0.5;markDirty();render();}});
window.addEventListener("beforeunload",event=>{if(state.dirty){event.preventDefault();event.returnValue="";}});
async function start(){$("annotator").value=localStorage.getItem("digit-annotator")||"";const response=await fetch("/api/samples",{cache:"no-store"});const payload=await response.json();state.samples=payload.samples;if(!state.samples.length)throw new Error("Annotation queue is empty");queueStatus(payload.status);const replacement=state.samples.findIndex(sample=>sample.replaces_sample_id&&!sample.saved&&!sample.rejected);const first=replacement>=0?replacement:state.samples.findIndex(sample=>!sample.saved&&!sample.rejected);await loadSample(first>=0?first:0);}
start().catch(error=>status(error.message,"error"));
