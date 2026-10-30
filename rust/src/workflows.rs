//! Typed workflow construction and media/loader policy for all remote encoders.
use crate::protocol::RequestId;
use crate::{from_python, to_python};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use serde_json::{Value, json};
fn invalid(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}
fn runtime(message: impl Into<String>) -> PyErr {
    PyRuntimeError::new_err(message.into())
}
fn text<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}
fn nonempty(v: &Value, key: &str) -> bool {
    text(v, key).is_some_and(|s| !s.trim().is_empty() && !s.contains('\0'))
}
fn array<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    v.get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}
fn link(name: &str, slot: u32) -> Value {
    json!([name, slot])
}
fn node(class: &str, inputs: Value) -> Value {
    json!({"class_type":class,"inputs":inputs})
}
struct ClipSelection {
    name: String,
    kind: String,
    device: String,
}
impl ClipSelection {
    fn parse(v: &Value) -> PyResult<Self> {
        let c = array(v, "clip");
        if c.len() != 3
            || c.iter()
                .any(|s| s.as_str().is_none_or(|s| s.trim().is_empty()))
        {
            return Err(invalid(
                "Remote Conditioning CLIP selection requires non-empty clip_name, clip_type, and device",
            ));
        }
        if c.iter().any(|s| s.as_str().unwrap().contains('\0')) {
            return Err(invalid(
                "Remote Conditioning CLIP selection contains a null byte",
            ));
        }
        Ok(Self {
            name: c[0].as_str().unwrap().into(),
            kind: c[1].as_str().unwrap().into(),
            device: c[2].as_str().unwrap().into(),
        })
    }
    fn node(self) -> Value {
        node(
            "CLIPLoader",
            json!({"clip_name": self.name, "type": self.kind, "device": self.device}),
        )
    }
}
#[derive(Clone, Copy, PartialEq)]
enum ConditioningStrategy {
    Text,
    Grounded,
}
impl ConditioningStrategy {
    fn parse(name: &str) -> PyResult<Self> {
        match name {
            "text" => Ok(Self::Text),
            "krea2-grounded" => Ok(Self::Grounded),
            _ => Err(invalid(format!(
                "Remote Conditioning strategy is not registered: {name:?}"
            ))),
        }
    }
    fn bounds(self) -> (usize, usize) {
        match self {
            Self::Text => (0, 0),
            Self::Grounded => (1, 2),
        }
    }
}
struct Dimensions {
    width: i64,
    height: i64,
    length: i64,
}
impl Dimensions {
    fn parse(v: &Value, reference: bool) -> PyResult<Self> {
        let width = v["width"].as_i64().unwrap_or(0);
        let height = v["height"].as_i64().unwrap_or(0);
        let length = v["length"].as_i64().unwrap_or(0);
        if width < 32
            || height < 32
            || length < 5
            || (!reference && (width % 32 != 0 || height % 32 != 0))
        {
            return Err(invalid(if reference {
                "Remote MiniMax H3 dimensions or ref_image_size are invalid"
            } else {
                "Remote MiniMax H3 Image to Video dimensions are invalid"
            }));
        }
        Ok(Self {
            width,
            height,
            length,
        })
    }
}
#[derive(Clone, Copy)]
enum Memory {
    Release,
    Keep,
}
impl Memory {
    fn parse(v: &Value, prefix: &str) -> PyResult<Self> {
        let value = match v.get("peer_memory_policy") {
            None => "release_after_result",
            Some(value) => value.as_str().unwrap_or(""),
        };
        match value {
            "release_after_result" => Ok(Self::Release),
            "keep" => Ok(Self::Keep),
            _ => Err(invalid(format!(
                "{prefix} peer_memory_policy must be one of: release_after_result, keep"
            ))),
        }
    }
    fn apply(self, graph: &mut Value, after: &str) {
        if matches!(self, Self::Release) {
            graph["cleanup"] = node(
                "VRAMCleanup",
                json!({"anything":link(after,0),"offload_model":true,"offload_cache":true,"reset_execution_cache":false}),
            );
        } else if let Some(g) = graph.as_object_mut() {
            g.remove("cleanup");
        }
    }
}
fn scope<'a>(v: &'a Value, prefix: &str) -> PyResult<&'a str> {
    let request = text(v, "request_id").unwrap_or("");
    if RequestId::parse(request).is_none() {
        return Err(invalid(format!(
            "{prefix} request_id must be 1-64 URL-safe characters"
        )));
    }
    Ok(request)
}
fn batch(graph: &mut Value, prefix: &str, refs: &[Value], empty: &str) -> PyResult<Value> {
    if refs.is_empty() {
        return Err(invalid(empty));
    }
    let mut previous = Value::Null;
    for (i, r) in refs.iter().enumerate() {
        let name = format!("{prefix}_{i}");
        graph[&name] = node("LoadImage", json!({"image":r}));
        let current = link(&name, 0);
        if i == 0 {
            previous = current;
        } else {
            let name = format!("{prefix}_batch_{i}");
            graph[&name] = node("ImageBatch", json!({"image1":previous,"image2":current}));
            previous = link(&name, 0);
        }
    }
    Ok(previous)
}
fn conditioning(v: &Value, template: Option<Value>) -> PyResult<Value> {
    let memory = Memory::parse(v, "Remote Conditioning")?;
    let clip_node = ClipSelection::parse(v)?.node();
    let request = scope(v, "Remote Conditioning")?;
    let prompt = text(v, "prompt")
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| invalid("Remote Conditioning prompt must be a non-empty string"))?;
    let strategy = text(v, "strategy_name").unwrap_or("text");
    let selected = ConditioningStrategy::parse(strategy)?;
    let images = array(v, "image_refs");
    let (minimum, maximum) = selected.bounds();
    if images.len() < minimum || images.len() > maximum {
        return Err(invalid(format!(
            "Remote Conditioning {strategy} requires {minimum} to {maximum} images; got {}",
            images.len()
        )));
    }
    if selected == ConditioningStrategy::Text {
        let raw = template.ok_or_else(|| runtime("Remote Conditioning template is malformed"))?;
        let mut graph = if let Some(path) = raw.as_str() {
            let bytes = std::fs::read(path)
                .map_err(|_| runtime(format!("Remote Conditioning template is invalid: {path}")))?;
            serde_json::from_slice(&bytes)
                .map_err(|_| runtime(format!("Remote Conditioning template is invalid: {path}")))?
        } else {
            raw
        };
        for path in [
            "/clip/inputs",
            "/encode/inputs/text",
            "/save/inputs/filename_prefix",
        ] {
            if graph.pointer(path).is_none() {
                return Err(runtime("Remote Conditioning template is malformed"));
            }
        }
        let filename = graph["save"]["inputs"]["filename_prefix"]
            .as_str()
            .ok_or_else(|| runtime("Remote Conditioning template is malformed"))?;
        if graph["clip"]["inputs"]["clip_name"] != "__REMOTE_CONDITIONING_CLIP_NAME__"
            || graph["clip"]["inputs"]["type"] != "__REMOTE_CONDITIONING_CLIP_TYPE__"
            || graph["clip"]["inputs"]["device"] != "__REMOTE_CONDITIONING_CLIP_DEVICE__"
            || graph["encode"]["inputs"]["text"] != "__REMOTE_CONDITIONING_TEXT__"
            || filename
                .matches("__REMOTE_CONDITIONING_REQUEST_ID__")
                .count()
                != 1
        {
            return Err(runtime(
                "Remote Conditioning template has unexpected injection markers",
            ));
        }
        let filename = filename.replace("__REMOTE_CONDITIONING_REQUEST_ID__", request);
        graph["clip"]["inputs"] = clip_node["inputs"].clone();
        graph["encode"]["inputs"]["text"] = prompt.into();
        graph["save"]["inputs"]["filename_prefix"] = filename.into();
        memory.apply(&mut graph, "save");
        return Ok(graph);
    }
    if v.get("grounding_px")
        .and_then(Value::as_u64)
        .is_none_or(|n| n > 4096)
    {
        return Err(invalid(
            "Remote Conditioning grounding_px must be 0 to 4096",
        ));
    }
    if text(v, "system_prompt").is_none() {
        return Err(invalid(
            "Remote Conditioning system_prompt must be a string",
        ));
    }
    if images
        .iter()
        .any(|s| s.as_str().is_none_or(|s| !s.ends_with(" [temp]")))
    {
        return Err(invalid(
            "Remote Conditioning image references must use validated temp uploads",
        ));
    }
    let mut graph = json!({"clip":clip_node,"encode":node("Krea2EditGroundedEncode",json!({"clip":link("clip",0),"prompt":prompt,"image":link("image_1",0),"grounding_px":v["grounding_px"],"system_prompt":v["system_prompt"]})),"save":node("SaveConditioning",json!({"conditioning":link("encode",0),"filename_prefix":format!("remote-conditioning/{request}/cond")}))});
    for (i, r) in images.iter().enumerate() {
        graph[format!("image_{}", i + 1)] = node("LoadImage", json!({"image":r}));
    }
    if images.len() == 2 {
        graph["encode"]["inputs"]["image_b"] = link("image_2", 0);
    }
    memory.apply(&mut graph, "save");
    Ok(graph)
}
fn qwen(v: &Value) -> PyResult<Value> {
    let memory = Memory::parse(v, "Remote Conditioning")?;
    let images = array(v, "image_refs");
    if images.len() > 16 {
        return Err(invalid("Remote Qwen Image 2.1 accepts at most 16 images"));
    }
    if v.get("resolution")
        .and_then(Value::as_u64)
        .is_none_or(|n| n > 4096)
    {
        return Err(invalid(
            "Remote Qwen Image 2.1 resolution must be 0 to 4096",
        ));
    }
    for name in ["prompt", "negative_prompt", "vae_name", "request_id"] {
        if name == "negative_prompt" && v[name] == "" {
            continue;
        }
        if !nonempty(v, name) {
            return Err(invalid(format!(
                "Remote Qwen Image 2.1 {name} must be a non-empty string"
            )));
        }
    }
    let request = scope(v, "Remote Qwen Image 2.1")?;
    let mut graph = json!({"clip":ClipSelection::parse(v)?.node(),"vae":node("VAELoader",json!({"vae_name":v["vae_name"]})),"encode":node("RemoteQwenImage21Encode",json!({"clip":link("clip",0),"vae":link("vae",0),"prompt":v["prompt"],"negative_prompt":v["negative_prompt"],"resolution":v["resolution"],"negative_vision":v["negative_vision"].as_bool().unwrap_or(false)}))});
    for (name, field, slot, prior, suffix) in [
        ("save_positive", "conditioning", 0, None, "positive"),
        (
            "save_negative",
            "conditioning",
            1,
            Some("save_positive"),
            "negative",
        ),
        ("save_latent", "samples", 2, Some("save_negative"), "latent"),
    ] {
        let mut inputs = json!({field:link("encode",slot),"filename_prefix":format!("remote-conditioning/{request}/{suffix}")});
        if let Some(prior) = prior {
            inputs["dependency"] = link(prior, 0);
        }
        graph[name] = node(
            if field == "samples" {
                "SaveRemoteLatent"
            } else {
                "SaveRemoteConditioning"
            },
            inputs,
        );
    }
    for (i, r) in images.iter().enumerate() {
        let name = format!("image_{}_loader", i + 1);
        graph[&name] = node("LoadImage", json!({"image":r}));
        graph["encode"]["inputs"][format!("image_{}", i + 1)] = link(&name, 0);
    }
    memory.apply(&mut graph, "save_latent");
    Ok(graph)
}
fn minimax(v: &Value, reference: bool) -> PyResult<Value> {
    let memory = Memory::parse(v, "Remote MiniMax H3")?;
    if !nonempty(v, "vae_name")
        || !nonempty(v, "prompt")
        || reference && !nonempty(v, "audio_vae_name")
    {
        return Err(invalid(if reference {
            "Remote MiniMax H3 requires non-empty VAE names and prompt"
        } else {
            "Remote MiniMax H3 Image to Video requires a non-empty prompt and VAE name"
        }));
    }
    let Dimensions {
        width,
        height,
        length,
    } = Dimensions::parse(v, reference)?;
    if reference && !matches!(text(v, "ref_image_size"), Some("match" | "max")) {
        return Err(invalid(
            "Remote MiniMax H3 dimensions or ref_image_size are invalid",
        ));
    }
    let request = scope(v, "Remote MiniMax H3")?;
    let mut encode = json!({"clip":link("clip",0),"vae":link("vae",0),"prompt":v["prompt"],"width":width,"height":height,"length":length});
    let mut graph = json!({"clip":ClipSelection::parse(v)?.node(),"vae":node("VAELoader",json!({"vae_name":v["vae_name"]})),"save":node("SaveRemoteConditioning",json!({"conditioning":link("encode",0),"filename_prefix":format!("remote-conditioning/{request}/cond")})),"save_latent":node("SaveRemoteLatent",json!({"samples":link("encode",1),"filename_prefix":format!("remote-conditioning/{request}/latent")}))});
    if reference {
        let images = array(v, "image_refs");
        let videos = array(v, "video_refs");
        let va = array(v, "video_audio_refs");
        let audios = array(v, "audio_refs");
        if images.len() > 9 || videos.len() > 3 {
            return Err(invalid("Remote MiniMax H3 reference count is invalid"));
        }
        if va.len() > 3 || audios.len() > 3 {
            return Err(invalid(
                "Remote MiniMax H3 audio reference count is invalid",
            ));
        }
        if va.len() > videos.len() {
            return Err(invalid(
                "Remote MiniMax H3 video audio cannot exceed reference videos",
            ));
        }
        graph["audio_vae"] = node("VAELoader", json!({"vae_name":v["audio_vae_name"]}));
        encode["audio_vae"] = link("audio_vae", 0);
        encode["ref_image_size"] = v["ref_image_size"].clone();
        for (i, r) in images.iter().enumerate() {
            let name = format!("ref_image_{i}");
            graph[&name] = node("LoadImage", json!({"image":r}));
            encode[format!("ref_images.{name}")] = link(&name, 0);
        }
        for (i, frames) in videos.iter().enumerate() {
            let refs = frames
                .as_array()
                .ok_or_else(|| invalid("Remote MiniMax H3 reference video has no frames"))?;
            let prefix = format!("ref_video_{i}_frame");
            let mut previous = Value::Null;
            for (fi, frame) in refs.iter().enumerate() {
                let name = format!("{prefix}_{fi}");
                graph[&name] = node("LoadImage", json!({"image":frame}));
                let current = link(&name, 0);
                if fi == 0 {
                    previous = current;
                } else {
                    let name = format!("ref_video_{i}_batch_{fi}");
                    graph[&name] = node("ImageBatch", json!({"image1":previous,"image2":current}));
                    previous = link(&name, 0);
                }
            }
            if refs.is_empty() {
                return Err(invalid("Remote MiniMax H3 reference video has no frames"));
            }
            encode[format!("ref_videos.ref_video_{i}")] = previous;
        }
        for (values, prefix, group) in [
            (va, "ref_video_audio", "ref_video_audios"),
            (audios, "ref_audio", "ref_audios"),
        ] {
            for (i, r) in values.iter().enumerate() {
                let name = format!("{prefix}_{i}");
                graph[&name] = node("LoadAudio", json!({"audio":r}));
                encode[format!("{group}.{name}")] = link(&name, 0);
            }
        }
    } else {
        for field in ["first_frame", "last_frame"] {
            if !v[field].is_null() {
                let name = format!("{field}_loader");
                graph[&name] = node("LoadImage", json!({"image":v[field]}));
                encode[field] = link(&name, 0);
            }
        }
    }
    graph["encode"] = node(
        if reference {
            "MiniMaxH3ReferenceToVideo"
        } else {
            "MiniMaxH3ImageToVideo"
        },
        encode,
    );
    if matches!(memory, Memory::Release) {
        graph["save_latent"]["inputs"]["dependency"] = link("save", 0);
    }
    memory.apply(&mut graph, "save_latent");
    Ok(graph)
}
fn prompt_workflow(v: &Value) -> PyResult<Value> {
    let request = scope(v, "Remote Prompt Enhancer")?;
    let p = &v["profile"];
    if !p.is_object() {
        return Err(invalid("Remote Prompt Enhancer profile is invalid"));
    }
    let images = array(v, "image_refs");
    let videos = array(v, "video_refs");
    let media = array(p, "accepted_media");
    let id = text(p, "id").unwrap_or("");
    if !images.is_empty() && !media.contains(&json!("image")) {
        return Err(invalid(format!(
            "Remote Prompt Enhancer profile {id} does not accept images"
        )));
    }
    if !videos.is_empty() && !media.contains(&json!("video")) {
        return Err(invalid(format!(
            "Remote Prompt Enhancer profile {id} does not accept video"
        )));
    }
    let mut graph = json!({"clip":node("CLIPLoader",json!({"clip_name":p["clip_name"],"type":p["clip_type"],"device":p["device"]})),"generate":node("RemotePromptGenerate",json!({"clip":link("clip",0),"prompt":v["prompt"],"system_prompt":p["system_prompt"],"media_strategy":if p["clip_type"]=="minimax"{"minimax-ref2va"}else{"qwen-image"},"max_length":p["max_length"]})),"save":node("SaveRemoteText",json!({"text":link("generate",0),"filename_prefix":format!("remote-conditioning/{request}/prompt")}))});
    if !images.is_empty() {
        graph["generate"]["inputs"]["image"] = batch(
            &mut graph,
            "image",
            images,
            "Remote Prompt Enhancer image batch cannot be empty",
        )?;
    }
    for (i, frames) in videos.iter().enumerate() {
        let refs = frames.as_array().map(Vec::as_slice).unwrap_or(&[]);
        graph["generate"]["inputs"][format!("video_{i}")] = batch(
            &mut graph,
            &format!("video_{i}"),
            refs,
            "Remote Prompt Enhancer image batch cannot be empty",
        )?;
    }
    Ok(graph)
}
#[pyfunction]
#[pyo3(signature=(kind,payload,template=None))]
fn workflow_build(
    kind: &str,
    payload: &Bound<'_, PyAny>,
    template: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let v = from_python(payload)?;
    let graph = match kind {
        "conditioning" => conditioning(&v, template.map(from_python).transpose()?)?,
        "qwen" => qwen(&v)?,
        "minimax-reference" => minimax(&v, true)?,
        "minimax-i2v" => minimax(&v, false)?,
        "prompt" => prompt_workflow(&v)?,
        _ => return Err(invalid("Unknown remote workflow kind")),
    };
    to_python(payload.py(), &graph)
}
#[pyfunction]
fn generated_prompt(
    prompt: &Bound<'_, PyAny>,
    system_prompt: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let prompt = prompt
        .extract::<String>()
        .ok()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| invalid("Remote Prompt Enhancer prompt must be a non-empty string"))?;
    if prompt.contains('\0') || prompt.len() > 128 * 1024 {
        return Err(invalid(
            "Remote Prompt Enhancer prompt is invalid or too large",
        ));
    }
    let system = system_prompt
        .extract::<String>()
        .ok()
        .filter(|s| !s.trim().is_empty() && !s.contains('\0'))
        .ok_or_else(|| invalid("Remote Prompt Enhancer system_prompt is invalid"))?;
    Ok(format!(
        "{}\n\nUser request:\n{}",
        system.trim(),
        prompt.trim()
    ))
}
#[pyfunction]
fn decode_generated_prompt_native(data: &Bound<'_, PyAny>) -> PyResult<String> {
    if !data.is_instance_of::<pyo3::types::PyBytes>() {
        return Err(runtime(
            "Remote Prompt Enhancer returned an invalid prompt artifact",
        ));
    }
    let bytes: Vec<u8> = data.extract()?;
    if bytes.is_empty() || bytes.len() > 128 * 1024 {
        return Err(runtime(
            "Remote Prompt Enhancer returned an invalid prompt artifact",
        ));
    }
    let result = std::str::from_utf8(&bytes)
        .map_err(|_| runtime("Remote Prompt Enhancer returned a non-UTF-8 prompt artifact"))?
        .trim();
    if result.is_empty() || result.contains('\0') {
        return Err(runtime("Remote Prompt Enhancer returned an empty prompt"));
    }
    Ok(result.to_owned())
}
#[pyfunction]
fn workflow_validate(
    kind: &str,
    info: &Bound<'_, PyAny>,
    name: &str,
    image_count: usize,
) -> PyResult<()> {
    let v = from_python(info)?;
    let has = |path: &str, keys: &[&str]| {
        v.pointer(path).is_some_and(|v| {
            keys.iter()
                .all(|k| v.get(*k).is_some() || v.as_array().is_some_and(|a| a.contains(&json!(k))))
        })
    };
    match kind {
        "conditioning" => {
            let remote = if name == "text" {
                "CLIPTextEncode"
            } else {
                "Krea2EditGroundedEncode"
            };
            if v.pointer("/input/required").is_none() || v.get("output").is_none() {
                return Err(runtime(format!(
                    "Remote Conditioning {remote} capability is malformed"
                )));
            }
            if !has("/input/required", &["clip", "prompt"])
                || !has(
                    "/input/optional",
                    if image_count > 1 {
                        &["image", "image_b"]
                    } else {
                        &["image"]
                    },
                )
                || !has("/output", &["CONDITIONING"])
            {
                return Err(runtime(format!(
                    "Remote Conditioning peer {remote} is incompatible with {name}"
                )));
            }
        }
        "qwen" => {
            if v.pointer("/input/required").is_none() || v.get("output").is_none() {
                return Err(runtime(
                    "Remote Qwen Image 2.1 peer capability is malformed",
                ));
            }
            if !has(
                "/input/required",
                &["clip", "prompt", "negative_prompt", "resolution"],
            ) || !has("/input/optional", &["vae", "negative_vision", "image_1"])
                || !has("/output", &["CONDITIONING", "LATENT"])
            {
                return Err(runtime(
                    "Remote Qwen Image 2.1 peer is incompatible; update custom-nodes on the peer",
                ));
            }
        }
        "minimax-reference" => {
            if !has("/output", &["CONDITIONING", "LATENT"]) {
                return Err(runtime("Remote MiniMax H3 peer is incompatible"));
            }
            if v.pointer("/input/optional").is_none() {
                return Err(runtime(
                    "Remote MiniMax H3 peer ref2va capability is malformed",
                ));
            }
            if !has(
                "/input/optional",
                &["ref_images", "ref_videos", "ref_video_audios", "ref_audios"],
            ) {
                return Err(runtime(
                    "Remote MiniMax H3 peer does not support the nested reference input contract",
                ));
            }
        }
        _ => return Err(invalid("Unknown remote capability kind")),
    }
    Ok(())
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(workflow_build, m)?)?;
    m.add_function(wrap_pyfunction!(workflow_validate, m)?)?;
    m.add_function(wrap_pyfunction!(generated_prompt, m)?)?;
    m.add_function(wrap_pyfunction!(decode_generated_prompt_native, m)?)?;
    Ok(())
}
