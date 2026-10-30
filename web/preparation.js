import {app} from '../../scripts/app.js';
import {api} from '../../scripts/api.js';
import {schedulerText} from './scheduler_view.js';

const labels = {
  conditioning: '다음 인코딩 중', conditioning_ready: '인코딩 준비됨',
  conditioning_consumed: '준비 결과 사용', vae: 'VAE 준비 중', vae_ready: 'VAE 준비됨',
  vae_deferred: '다음 인코딩 우선', skipped: '실행 시 처리', deferred: '원격 대기',
  error: '자동 준비 보류', cancelled: '사전 계산 중단',
};
const reasons = {
  dynamic_inputs_or_refresh: '입력은 실행 시 계산',
  handoff_too_large: '준비 결과의 RAM 한도 초과 · 실행 시 다시 계산',
  queue_changed: '대기열 취소 또는 입력 변경', next_clip_priority: '다음 인코딩 우선',
  foreground_priority: '현재 작업 우선 · 다음 작업은 실행 시 계산',
  local_or_missing_peer: '별도 처리 서버 필요', peer_busy: '처리 서버 사용 중',
};

function text(tag, value, className) {
  const el = document.createElement(tag);
  el.textContent = value;
  if (className) el.className = className;
  return el;
}

export async function showPreparation() {
  const existing = document.querySelector('dialog.remote-preparation');
  if (existing) { existing.querySelector('button')?.focus(); return; }
  const opener = document.activeElement;
  const dialog = document.createElement('dialog');
  dialog.className = 'remote-preparation';
  dialog.setAttribute('aria-labelledby', 'remote-preparation-title');
  const title = text('h2', '준비 상태'); title.id = 'remote-preparation-title';
  const close = text('button', '닫기'); close.type = 'button'; close.onclick = () => dialog.close();
  const header = document.createElement('header'); header.append(title, close);
  const status = text('p', '상태 확인 중'); status.setAttribute('role', 'status');
  const schedule = document.createElement('section'); schedule.className = 'preparation-schedule';
  schedule.setAttribute('aria-label', '작업별 상태'); schedule.hidden = true;
  const policy = text('p', '', 'preparation-hint'); policy.hidden = true;
  const diagnostic = text('p', '', 'preparation-hint'); diagnostic.hidden = true;
  const content = document.createElement('div'); content.className = 'preparation-events';
  const jobDetails = document.createElement('section'); jobDetails.className = 'preparation-jobs'; jobDetails.hidden = true;
  const details = document.createElement('details'); details.className = 'preparation-details';
  details.append(text('summary', '상세·이력'), diagnostic, policy, jobDetails, content);
  const refresh = text('button', '새로고침'); refresh.type = 'button';
  dialog.append(header, status, schedule, details, refresh);
  let stopped = false, busy = false, pendingRequest = null, generation = 0;
  async function update() {
    if (stopped || busy || document.hidden) return;
    const request = ++generation;
    busy = true; refresh.disabled = true; refresh.textContent = '확인 중';
    dialog.setAttribute('aria-busy', 'true');
    const controller = new AbortController(); pendingRequest = controller;
    let timedOut = false;
    const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 10000);
    try {
      const response = await api.fetchApi('/remoteworker/preparation', {signal: controller.signal});
      if (!response.ok) throw new Error(`상태 조회 실패 (${response.status})`);
      const data = await response.json();
      if (stopped || request !== generation) return;
      if (!data || (!data.pipeline && !data.queue && !data.active)) throw new Error('준비 상태 응답이 없습니다.');
      const timing = schedulerText(data.scheduler, data.pipeline);
      status.textContent = timing?.title || (data.active ? labels[data.active.state] || '상태 확인 중'
        : data.queue?.running ? '현재 작업 실행 중' : '대기');
      diagnostic.hidden = true;
      jobDetails.hidden = !timing?.jobs?.length;
      jobDetails.replaceChildren(text('strong', '작업 ID'), ...(timing?.jobs || []).map(job => text('code', `${job.id} · ${job.label}`)));
      policy.textContent = timing?.description || '';
      policy.hidden = !policy.textContent;
      if (timing?.jobs?.length) {
        const list = document.createElement('ul');
        for (const job of timing.jobs) {
          const row = document.createElement('li');
          row.append(text('code', job.id.slice(0, 8)), text('span', job.label));
          list.append(row);
        }
        schedule.replaceChildren(list); schedule.hidden = false;
      } else if (timing?.timing) {
        schedule.replaceChildren(text('p', timing.timing)); schedule.hidden = false;
      } else { schedule.replaceChildren(); schedule.hidden = true; }
      content.replaceChildren(...(Array.isArray(data.events) ? data.events : []).slice(-8).reverse().map(event => {
        const row = document.createElement('section');
        row.append(text('strong', labels[event.state] || event.state));
        const detail = [event.prompt_id, event.node_id ? `노드 ${event.node_id}` : '', event.model,
          reasons[event.reason] || event.reason].filter(Boolean).join(' · ');
        if (detail) row.append(text('p', detail));
        if (Number.isFinite(event.at)) row.append(text('time', new Date(event.at * 1000).toLocaleTimeString('ko-KR')));
        return row;
      }));
      refresh.textContent = '새로고침';
    } catch (error) {
      if (!stopped && request === generation) {
        status.textContent = timedOut ? '응답 지연 · 다시 확인' : '상태 확인 실패 · 다시 확인';
        schedule.hidden = true;
        if (!jobDetails.hidden) jobDetails.querySelector('strong').textContent = '마지막 확인 작업';
        diagnostic.textContent = error.message; diagnostic.hidden = !error.message;
        refresh.textContent = '다시 확인';
      }
    } finally {
      clearTimeout(timeout);
      if (!stopped && request === generation) {
        pendingRequest = null; busy = false; refresh.disabled = false;
        dialog.setAttribute('aria-busy', 'false');
      }
    }
  }
  refresh.onclick = update;
  const timer = setInterval(update, 2000);
  dialog.addEventListener('close', () => {
    stopped = true; generation++; pendingRequest?.abort(); clearInterval(timer);
    dialog.remove(); if (opener?.isConnected) opener.focus();
  });
  document.body.append(dialog); dialog.showModal(); close.focus(); await update();
}

app.registerExtension({
  name: 'RemoteWorker.AutomaticPreparation',
  nodeCreated(node) {
    if (!['RemoteVAEDecode', 'RemoteVAEDecodeAudio'].includes(node.comfyClass || node.type)) return;
    const button = node.addWidget('button', '준비 상태', null, showPreparation, {serialize: false});
    button.serialize = false;
    if (node.computeSize) node.setSize?.([node.size?.[0] || 360, node.computeSize()[1]]);
  },
  async setup() {
    if (!document.getElementById('remote-preparation-style')) {
      const css = document.createElement('link'); css.id = 'remote-preparation-style'; css.rel = 'stylesheet';
      css.href = new URL('./preparation.css', import.meta.url).href; document.head.append(css);
    }
    app.extensionManager.registerSidebarTab({
      id: 'remote-preparation', icon: 'pi pi-bolt', title: '준비 상태', tooltip: '자동 준비 상태',
      type: 'custom', render: el => {
        el.replaceChildren(); el.classList.add('remote-preparation-sidebar');
        const button = text('button', '준비 상태'); button.type = 'button'; button.onclick = showPreparation;
        el.append(button);
      },
    });
  },
});
