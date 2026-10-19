import {fetchJson} from './remote_choices.js';

const types = new Set(['RemoteMiniMaxH3ReferenceEncode', 'RemoteMiniMaxH3ImageToVideo']);
const discovered = new Map();
const basename = value => String(value || '').replaceAll('\\', '/').split('/').pop();

export function applyMiniMaxModelDefinitions(definitions) {
  for (const definition of definitions || []) {
    if (!types.has(definition.name)) continue;
    const required = definition.input?.required;
    const peer = required?.server_url?.[1]?.default;
    const models = discovered.get(peer);
    if (!models) continue;
    for (const field of ['clip_name', 'vae_name', 'audio_vae_name']) {
      const spec = required[field];
      if (!Array.isArray(spec?.[0])) continue;
      const values = field === 'clip_name' ? models.clip : models.vae;
      const previous = spec[1]?.default || spec[0][0];
      const selected = values.includes(previous) ? previous : values.find(value => basename(value) === basename(previous));
      spec[0].splice(0, spec[0].length, ...values);
      spec[1] = {...spec[1], default: selected || previous};
    }
  }
}

export async function discoverMiniMaxModelDefinitions(definitions) {
  const list = Object.values(definitions || {});
  const peers = [...new Set(list.filter(definition => types.has(definition.name))
    .map(definition => definition.input?.required?.server_url?.[1]?.default).filter(Boolean))];
  await Promise.all(peers.map(async peer => {
    try {
      const params = {server_url: peer, timeout_seconds: 5};
      const [clip, vae] = await Promise.all([
        fetchJson('/remote-conditioning/clip-loader-options', params),
        fetchJson('/remote-conditioning/vae-loader-options', params),
      ]);
      if (Array.isArray(clip.clip_name) && clip.clip_name.length && Array.isArray(vae.vae_name) && vae.vae_name.length)
        discovered.set(peer, {clip: clip.clip_name, vae: vae.vae_name});
    } catch (error) {
      console.warn('[RemoteWorker] Remote model definitions unavailable', error);
    }
  }));
  applyMiniMaxModelDefinitions(list);
}
