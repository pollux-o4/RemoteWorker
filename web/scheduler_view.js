const messages = {
  clip_required: ['다음 CLIP 우선 처리', '다음 conditioning을 완료한 뒤 VAE를 준비합니다.'],
  clip_window: ['CLIP 처리 구간', '경계 전까지 VAE 웜업을 보류합니다.'],
  sampling_progress: ['샘플링 진행 확인 중', '진행 step을 확인한 뒤 VAE 준비 시점을 결정합니다.'],
  vae_window: ['VAE 준비 구간', '종료 준비 구간 · 미처리 CLIP이 들어오면 먼저 처리합니다.'],
  vae_ready: ['VAE 준비 완료', '대기 작업의 conditioning이 없으면 CLIP을 먼저 처리한 뒤 VAE로 전환합니다.'],
};

export function schedulerText(scheduler, pipeline) {
  if (pipeline?.enabled) {
    const jobs=pipeline.jobs || [];
    const remote=jobs.filter(job => job.phase === 'remote');
    const local=jobs.find(job => job.prompt_id === pipeline.local_prompt_id);
    const ready=jobs.some(job => job.phase === 'ready');
    const previewed=jobs.some(job => job.preview_nodes?.length);
    const title=pipeline.admission === 'ram_pressure' ? '메모리 확보 대기'
      : ready ? (previewed ? '미리보기 표시 · 후속 처리 대기' : '디코딩 완료 · 후속 처리 대기')
      : local && remote.length ? '다음 작업·원격 디코딩 병행'
      : remote.length ? '원격 디코딩 중' : local ? '모델 준비·샘플링 처리 중' : '파이프라인 대기';
    return {title, description:'원격 디코딩 중 다음 작업 진행 · 미리보기는 원래 작업에 표시',
      timing:jobs.map(job => `${job.prompt_id.slice(0,8)} · ${job.phase === 'remote' ? '원격 처리' : job.phase === 'ready' ? (job.preview_nodes?.length ? '미리보기 표시 · 후속 대기' : '결과 준비 완료') : '로컬 처리'}`).join(' / ')};
  }
  const decision=scheduler?.decision;
  if (!decision?.reason) return null;
  const [title,description]=messages[decision.reason] || ['작업 상태 확인 중',''];
  const progress=scheduler.sampling;
  const remaining=progress?.prompt_id===decision.prompt_id && Number.isFinite(progress?.total) && Number.isFinite(progress?.step)
    ? Math.max(0,progress.total-progress.step) : decision.remaining_steps;
  const timing=Number.isFinite(remaining) ? `샘플링 ${remaining} step 남음` : '';
  return {title,description,timing};
}
