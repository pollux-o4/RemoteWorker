const messages = {
  clip_required: ['다음 CLIP 우선 처리', '다음 인코딩을 먼저 처리합니다.'],
  clip_window: ['샘플링 중', 'VAE 준비 시점을 기다립니다.'],
  sampling_progress: ['진행 확인 중', '샘플링 진행을 확인한 뒤 VAE를 준비합니다.'],
  vae_window: ['VAE 준비 중', '다음 인코딩이 들어오면 CLIP을 먼저 처리합니다.'],
  vae_ready: ['VAE 준비됨', '다음 인코딩이 있으면 CLIP을 먼저 처리합니다.'],
};

const phases = {
  local: '로컬 처리', remote: '원격 처리', ready: '로컬 복귀 대기',
  restoring: '상태 복원 중', failed: '실행 중단',
};

export function schedulerText(scheduler, pipeline) {
  if (pipeline?.enabled) {
    const jobs = (pipeline.jobs || []).map(job => ({
      id: String(job.prompt_id || ''),
      phase: job.phase,
      label: job.phase === 'ready' && job.preview_nodes?.length
        ? '미리보기 · 복귀 대기' : phases[job.phase] || '상태 확인 중',
    }));
    const remote = jobs.some(job => job.phase === 'remote');
    const local = jobs.some(job => job.id === pipeline.local_prompt_id);
    const title = pipeline.admission === 'ram_pressure' ? '메모리 대기'
      : jobs.some(job => job.phase === 'failed') ? '작업 중단'
      : jobs.some(job => job.phase === 'restoring') ? '상태 복원 중'
      : jobs.some(job => job.phase === 'ready') ? '로컬 복귀 대기'
      : local && remote ? '로컬·원격 병행'
      : remote ? '원격 처리 중' : local ? '로컬 처리 중' : '대기';
    return {title, jobs,
      description: '원격 처리 중 로컬 작업 진행 · 미리보기는 원래 작업에 표시',
      timing: jobs.map(job => `${job.id.slice(0, 8)} · ${job.label}`).join(' / ')};
  }
  const decision = scheduler?.decision;
  if (!decision?.reason) return null;
  const [title, description] = messages[decision.reason] || ['상태 확인 중', ''];
  const progress = scheduler.sampling;
  const remaining = progress?.prompt_id === decision.prompt_id && Number.isFinite(progress?.total) && Number.isFinite(progress?.step)
    ? Math.max(0, progress.total - progress.step) : decision.remaining_steps;
  return {title, description,
    timing: Number.isFinite(remaining) ? `샘플링 ${remaining} step 남음` : ''};
}
