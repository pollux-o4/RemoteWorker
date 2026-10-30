import {app} from '../../scripts/app.js';
import {api} from '../../scripts/api.js';
import {schedulerText} from './scheduler_view.js';

const labels = {
  conditioning: '다음 인코딩',
  conditioning_ready: '인코딩 완료',
  conditioning_consumed: '준비 결과 사용',
  vae: 'VAE 준비',
  vae_ready: 'VAE 준비 완료',
  vae_deferred: '다음 CLIP으로 전환',
  skipped: '실행 시 처리',
  deferred: '처리 서버 대기',
  error: '자동 준비 보류',
  cancelled: '사전 계산 중단',
};
const reasons = {
  dynamic_inputs_or_refresh: '사전 계산할 수 없는 입력',
  handoff_too_large: '준비 결과의 RAM 한도 초과 · 실행 시 다시 처리',
  queue_changed: '대기열 취소 또는 입력 변경',
  next_clip_priority: '다음 conditioning 우선 처리 후 VAE 준비',
  foreground_priority: '현재 작업을 우선 처리 · 다음 작업은 실행 시 다시 계산',
  local_or_missing_peer: '별도 처리 서버 필요', peer_busy: '처리 서버가 다른 작업 실행 중',
};

function text(tag, value, className) {
  const el = document.createElement(tag);
  el.textContent = value;
  if (className) el.className = className;
  return el;
}

async function showPreparation() {
  const existing = document.querySelector('dialog.remote-preparation');
  if (existing) { existing.querySelector('button')?.focus(); return; }
  const opener = document.activeElement;
  const dialog = document.createElement('dialog');
  dialog.className = 'remote-preparation';
  dialog.setAttribute('aria-labelledby', 'remote-preparation-title');
  const title = text('h2', '자동 준비'); title.id = 'remote-preparation-title';
  const close = text('button', '×'); close.setAttribute('aria-label', '닫기'); close.title = '닫기'; close.onclick = () => dialog.close();
  const header = document.createElement('header'); header.append(title, close);
  const status = text('p', '상태 확인 중…'); status.setAttribute('role', 'status');
  const schedule = document.createElement('section'); schedule.className='preparation-schedule'; schedule.setAttribute('aria-label','다음 작업 스케줄'); schedule.hidden=true;
  const content = document.createElement('div'); content.className = 'preparation-events';
  const hint = text('p', '대기 작업의 conditioning이 없으면 CLIP을 먼저 처리합니다.', 'preparation-hint');
  const refresh = text('button', '새로고침');
  const details = document.createElement('details');
  details.className = 'preparation-details';
  details.append(text('summary', '상세'), hint, content);
  dialog.append(header, status, schedule, details, refresh);
  let stopped = false, busy = false, pendingRequest = null;
  async function update() {
    if (stopped || busy || document.hidden) return;
    busy = true;
    refresh.disabled = true;
    const controller = new AbortController(); pendingRequest = controller;
    const timeout = setTimeout(() => controller.abort(), 10000);
    try {
      const response = await api.fetchApi('/remoteworker/preparation', {signal: controller.signal});
      if (!response.ok) throw new Error('자동 준비 상태를 확인할 수 없습니다.');
      const data = await response.json();
      if (stopped) return;
      status.textContent = data.active ? labels[data.active.state] || data.active.state : data.queue?.running ? '현재 작업 실행 중' : '대기';
      if (data.pipeline?.enabled) hint.textContent="원격 디코딩 대기 중 다음 작업을 처리합니다. 다음 conditioning을 먼저 준비합니다.";
      else if (Number.isFinite(data.scheduler?.vae_remaining_steps)) hint.textContent=`${data.scheduler.vae_remaining_steps} step이 남으면 VAE를 준비합니다. 대기 작업의 conditioning이 없으면 CLIP을 먼저 처리합니다.`;
      const timing=schedulerText(data.scheduler,data.pipeline);
      schedule.hidden=!timing;
      if (timing) {
        schedule.title = timing.description;
        schedule.replaceChildren(text('strong',timing.title),text('p',timing.timing));
      }
      content.replaceChildren(...data.events.slice(-8).reverse().map(event => {
        const row = document.createElement('section');
        row.append(text('strong', labels[event.state] || event.state));
        const detail = [event.node_id ? `노드 ${event.node_id}` : '', event.model,
          reasons[event.reason] || event.reason].filter(Boolean).join(' · ');
        if (detail) row.append(text('p', detail));
        row.append(text('time', new Date(event.at * 1000).toLocaleTimeString('ko-KR')));
        return row;
      }));
    } catch (error) { if (!stopped) status.textContent = error.name === 'AbortError' ? '연결 응답 지연 · 다시 확인해 주세요.' : error.message; }
    finally { clearTimeout(timeout); pendingRequest = null; busy = false; refresh.disabled = false; }
  }
  refresh.onclick = update;
  const timer = setInterval(update, 2000);
  dialog.addEventListener('close', () => { stopped = true; pendingRequest?.abort(); clearInterval(timer); dialog.remove(); if (opener?.isConnected) opener.focus(); });
  document.body.append(dialog); dialog.showModal(); await update();
}

app.registerExtension({
  name: 'RemoteWorker.AutomaticPreparation',
  nodeCreated(node) {
    if (!['RemoteVAEDecode', 'RemoteVAEDecodeAudio'].includes(node.comfyClass || node.type)) return;
    const button = node.addWidget('button', '자동 준비 상태', null, showPreparation, {serialize: false});
    button.serialize = false;
    node.setSize?.([Math.max(node.size?.[0] || 0, 360), node.computeSize()[1]]);
  },
  async setup() {
    const css = document.createElement('link'); css.rel = 'stylesheet';
    css.href = new URL('./preparation.css', import.meta.url).href; document.head.append(css);
    app.extensionManager.registerSidebarTab({
      id: 'remote-preparation', icon: 'pi pi-bolt', title: '자동 준비', tooltip: 'RemoteWorker 자동 준비',
      type: 'custom', render: el => {
        el.replaceChildren(); el.classList.add('remote-preparation-sidebar');
        const button = text('button', '자동 준비 상태'); button.onclick = showPreparation;
        el.append(button);
      },
    });
  },
});
