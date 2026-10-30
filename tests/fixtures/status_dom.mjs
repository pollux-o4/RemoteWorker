export class Element {
  constructor(tag) { this.tagName=tag.toLowerCase();this.children=[];this.attributes=new Map();this.dataset={};this.hidden=false;this.listeners=new Map();this._text='';this.style={};this.disabled=false; }
  set textContent(value) { this._text=String(value);this.children=[]; }
  get textContent() { return this._text+this.children.map(child=>child.textContent).join(''); }
  get isConnected() { return this===document.body||this===document.head||Boolean(this.parentElement?.isConnected); }
  get scrollHeight() { return this._height??(20+this.children.filter(child=>!child.hidden).reduce((sum,child)=>sum+child.scrollHeight,0)); }
  get offsetHeight() { return this.scrollHeight; }
  setAttribute(name,value) { this.attributes.set(name,String(value));if(name==='id')this.id=String(value);if(name==='class')this.className=String(value);if(name.startsWith('data-'))this.dataset[name.slice(5).replace(/-([a-z])/g,(_,letter)=>letter.toUpperCase())]=String(value); }
  getAttribute(name) { return this.attributes.get(name)??null; }
  append(...children) { for(const child of children){if(child.parentElement)child.remove();child.parentElement=this;this.children.push(child);} }
  replaceChildren(...children) { for(const child of this.children)child.parentElement=null;this.children=[];this._text='';this.append(...children); }
  remove() { if(this.parentElement)this.parentElement.children=this.parentElement.children.filter(child=>child!==this);this.parentElement=null; }
  matches(selector) { const attribute=selector.match(/^\[([^=\]]+)(?:=["']?([^"'\]]+)["']?)?\]$/);if(attribute)return this.attributes.has(attribute[1])&&(attribute[2]===undefined||this.getAttribute(attribute[1])===attribute[2]);const [tag,cls]=selector.split('.');return (!tag||this.tagName===tag)&&(!cls||(this.className||'').split(' ').includes(cls)); }
  querySelectorAll(selector) { return this.children.flatMap(child=>[...(child.matches(selector)?[child]:[]),...child.querySelectorAll(selector)]); }
  querySelector(selector) { return this.querySelectorAll(selector)[0]??null; }
  addEventListener(type,listener) { this.listeners.set(type,[...(this.listeners.get(type)||[]),listener]); }
  dispatchEvent(event) { for(const listener of this.listeners.get(event.type)||[])listener.call(this,event);this['on'+event.type]?.call(this,event);return true; }
  click() { if(!this.disabled)this.dispatchEvent({type:'click'}); }
  focus() { document.activeElement=this; }
  showModal() { this.open=true;this.setAttribute('open',''); }
  close() { this.open=false;this.dispatchEvent({type:'close'}); }
  pause() { this.paused=true; }
  get classList() { return {add:name=>{this.className=[this.className,name].filter(Boolean).join(' ');}}; }
}
export function installDOM() {
  globalThis.CustomEvent=class {constructor(type,options={}){this.type=type;this.detail=options.detail;}};
  globalThis.document={head:new Element('head'),body:new Element('body'),activeElement:null,hidden:false,events:[],createElement:tag=>new Element(tag),
    getElementById(id){return [...this.head.querySelectorAll('link'),...this.body.querySelectorAll('h2')].find(el=>el.id===id)||null;},
    querySelector(selector){return this.body.querySelector(selector)||this.head.querySelector(selector);},dispatchEvent(event){this.events.push(event);return true;}};
  return document;
}
