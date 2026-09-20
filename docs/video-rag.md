# Video RAG

Every technique on this site so far treats a document as either text or a static image. Video adds a dimension neither has: time. A single frame or a fixed-length clip can miss context that only makes sense across a longer span — a graph being drawn stroke by stroke, an explanation that starts in one moment and lands in the next — and a query needs to find not just the right *content*, but the right *moment*.

!!! warning "Scope: this page builds the decomposed version, not native video understanding — read this before the rest"
    "Video RAG" spans three genuinely different things, and it matters which one you mean:

    1. **Decomposed retrieval, decomposed generation (what this page builds).** Speech-to-text produces a transcript; frames are sampled as still images; search only ever runs against the transcript text; generation gets one static frame plus transcript text, because the LLM used here (Claude) has no video input at all — its API accepts images and PDF documents, never a video file. This turns a video into a text-plus-image problem. It's real, it's what most practical open-source implementations actually do, and it's genuinely useful — but it is not a model perceiving video *as video*.
    2. **Native "omni" models that take actual video (frames + synchronized audio) as one input** — Gemini's video file API, Qwen2.5-Omni, GPT-4o's audio+vision path. Handed an actual clip, these can answer something like "what does he do right as he says X," which a single sampled frame plus separately-transcribed text structurally cannot, because motion and audio-visual timing were never decomposed away in the first place.
    3. **Video-native retrieval** — embedding whole clips directly (InternVideo, ViCLIP) instead of frame-by-frame CLIP, so even finding the right moment doesn't go through a text/image proxy (the approach in the VideoRAG paper, arXiv 2501.05874).

    This page is tier 1, on purpose: tiers 2 and 3 need either a model this recipe's generation step (Claude) cannot use, or video-embedding models in the 1B+ parameter range — both unrealistic on an 8GB machine without a dedicated GPU. Confirmed directly against Claude's own API docs before writing this: there is no video content-block type, only `image` and `document`. If you swap the generation step for a model with native video input, tier 1's whole "sample frames, pick the nearest one" mechanism becomes unnecessary — that's a materially different recipe, not a bigger version of this one.

!!! example "Hands-on"
    The full comparison below is runnable, against a real public-domain video: [**Video RAG →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/video-rag) in the code repo. Transcription and retrieval are fully local; only generation needs an API key.

??? abstract "TL;DR — quick revision"
    - **Two signals, one timestamp axis**: a speech-to-text transcript (Whisper) and frames sampled at a fixed interval, both indexed by time, so a query can retrieve the relevant moment's transcript AND see what was on screen at that moment
    - **Measured directly: transcript-only retrieval scores 3/5** on a real eval set built from this page's own sample video — both misses have a distinct, checked cause, not a vague "retrieval sometimes fails"
    - **One failure mirrors ordinary chunk-boundary problems, just on the time axis**: Whisper's own segmentation split a cause from its explanation across two adjacent segments — retrieval found the segment ending in "...begins to warm up because," but the actual reason was in the next one
    - **The other is embedding confusion between near-duplicates**: two adjacent segments about similar-sounding percentages (90% vs. 100% of two different UV bands) crowded each other out of the top-3
    - **There's a real, verified case transcript-only retrieval cannot solve at all**: a whiteboard axis label that's shown but never spoken — the transcript-only path correctly declines rather than guessing, and only passing the actual frame image gets the right answer
    - **This recipe deliberately doesn't index frames by visual similarity** — frame grounding here is timestamp-proximity to the retrieved transcript segment, not a separate CLIP-based visual search (that mechanism is already covered on the [Multimodal RAG](multimodal-rag.md) and [Vision RAG](vision-rag.md) pages)

## The sample video, and why it's a fair test

NASA's ["NASA Explains Cold Zone Above Tropics"](https://images.nasa.gov) (public domain, CC0) — a real, 5-minute-12-second video, verified directly before using it. NASA scientist Paul Newman explains atmospheric layers (troposphere, tropopause, stratosphere) and how the ozone layer absorbs UV radiation, while drawing a live graph on a whiteboard. The video actually mixes several visual styles — an animated title card, animated illustrations, hand-lettered word overlays, and genuine live-action whiteboard footage — and critically, the whiteboard's own axis labels ("ALTITUDE," "TEMPERATURE") are never spoken anywhere in the narration. That's what makes it a fair test: there's a real, verifiable fact that only exists visually, not just a hypothetical one.

## Pipeline: transcript and frames, both keyed by timestamp

Frames and transcript segments live on two different timelines, and they're linked by nothing more than arithmetic on a shared unit (seconds) — not a shared index, and not a joint embedding space.

**Frames sit on a fixed, even grid.** `ffmpeg` samples one frame every 5 seconds, so frame *i*'s timestamp is always `i × 5` — frame 0 at 0s, frame 1 at 5s, frame 2 at 10s, and so on, no exceptions, for the whole video.

**Transcript segments sit on an uneven grid Whisper decides on its own.** Segments are cut wherever Whisper detects a natural speech boundary — a pause, a sentence end — so their start times almost never land on a multiple of 5. A real segment from this video: `{"start": 94.88, "end": 100.44, "text": "This cold point is called the tropopause."}`.

**Only the transcript is searched.** The query is embedded once and compared by cosine similarity against every transcript segment's embedding — frames are never embedded against the query and never take part in the similarity search at all. The top-k result is just transcript segments, each carrying whatever `start`/`end` Whisper gave it.

**Frames are matched to a search result afterward, by nearest timestamp — nothing more.** Once a segment wins the search (say, the tropopause segment starting at 94.88s), the retrieved frame is whichever of the fixed 5-second samples is numerically closest to that start time:

```
segment start = 94.88s
frame timestamps = 0, 5, 10, ..., 90, 95, 100, ...
|94.88 - 90| = 4.88
|94.88 - 95| = 0.12   <- smallest, so the frame at 95s wins
```

That's the entire mechanism — `min(frames, key=lambda pair: abs(pair[0] - timestamp))`. The real consequence of it being this simple: since a frame exists only every 5 seconds, the one you get back is never guaranteed to be the exact right visual instant, just whichever sample happened to land closest. This worked for the whiteboard question below because the whiteboard stayed on screen for a while — any frame within a couple of seconds of it still showed the graph. A visual detail that only existed for a 1-2 second window between two samples could be missed entirely.

**Frame extraction** samples one frame every 5 seconds via `ffmpeg` — fixed-interval, not scene-detection, which is simple and adequate for a short video:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/video-rag/video_rag_docs.py:extract_frames"
```

**Transcription** produces timestamped segments. Whisper shells out to the system `ffmpeg` binary internally to decode audio, so it's a real prerequisite even though this function never calls it directly:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/video-rag/video_rag_docs.py:transcribe_video"
```

**Retrieval** over transcript segments uses the same embed-and-cosine-similarity pattern as plain text RAG elsewhere on this site — the only difference is each retrievable unit carries a timestamp instead of a page or chunk id:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/video-rag/video_rag_docs.py:transcript_search"
```

**Frame grounding** is driven entirely by proximity to the retrieved segment's timestamp — frames only exist at the fixed 5-second samples, so this is always "the closest sampled frame," not a guarantee of the exact right visual instant:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/video-rag/video_rag_docs.py:nearest_frame"
```

## Measured: transcript-only Recall@3 — 3 of 5

A 5-question eval set built from this video's own real transcript:

```text
PASS  What temperature range is found at ground level in the tropics?
PASS  What is the point called where the troposphere meets the stratosphere?
PASS  What percentage of UV-B radiation is absorbed by the ozone layer?
FAIL  What percentage of UV-C radiation is absorbed by the ozone layer?
FAIL  Why does the stratosphere begin to warm up?
```

!!! warning "Two misses, two distinct, checked causes"
    **UV-C question.** The exact right segment exists in the transcript — *"There's another class called UVC... 100% of the UVC"* — but it wasn't in the top-3. Two adjacent segments about UV-B's 90% figure and generic "absorbed by the ozone layer" phrasing crowded it out: three segments about percentages and ozone absorption look extremely similar to an embedding model, and the one with the actually-different number lost.

    **Stratosphere-warming question.** Retrieval correctly found the segment ending in *"...the reason the stratosphere begins to warm up is because"* — and stopped there, because that's where Whisper's own segmentation drew the line. The actual reason (*"...radiation from the Sun is being absorbed in the ozone layer"*) landed in the very next segment, which wasn't retrieved. This is the same shape as a text-chunking boundary problem — a cause and its explanation split across two retrieval units — just happening on the time axis instead of the character-count axis.

## The case transcript-only retrieval cannot answer

Asked *"What does the Y-axis of the whiteboard graph say?"* — a detail that's shown on screen but genuinely never spoken:

```text
Transcript-only answer: The transcript doesn't specify what the Y-axis of
the whiteboard graph says.

With-frame answer: Based on the image, the Y-axis of the whiteboard graph
appears to say "ALTITUDE" (written vertically along the left side of the
graph).
```

The transcript-only path doesn't hallucinate a plausible-sounding label — it correctly says the answer isn't in the given context. That's the right behavior with only text to work from. Passing the actual frame, retrieved by proximity to the closest relevant transcript segment, is what makes the question answerable at all. This is the direct, video-specific version of the same principle demonstrated on the [Multimodal RAG](multimodal-rag.md) page: some answers are visual facts, not textual ones, and no amount of better text retrieval recovers information that was never in the text to begin with.

## What this recipe leaves out, on purpose

There's no separate visual (CLIP-based) frame search here — frame grounding is timestamp proximity to whichever transcript segment retrieval already found, not a query embedded against frame images directly. A fuller pipeline could add that: embed each frame with a CLIP-family model (exactly the mechanism already built and measured on the [Multimodal RAG](multimodal-rag.md) and [Vision RAG](vision-rag.md) pages) to catch moments where the relevant visual doesn't line up with any well-matching transcript segment at all — a purely visual event with no narration over it, for instance. Left out here to keep this page focused on the timestamp-linking mechanism that's specific to video, rather than re-deriving image-embedding choices already covered elsewhere on this site.

## When to use which

**Transcript-only retrieval is enough when:** the answer is something that gets said out loud, and the video's narration reliably covers its own content — most explainer/lecture-style videos. **Add frame grounding when:** the video shows things it doesn't narrate — on-screen labels, diagrams, text, or visual detail that matters to the likely questions. **Consider a full CLIP-based visual search (not built here) when:** a meaningful fraction of the video's important content has no narration over it at all, so timestamp-proximity to a transcript hit isn't a reliable way to find the right frame.

## Scenario Check

<div class="quiz-widget" data-title="Scenario Check: Video RAG">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team builds transcript-only video retrieval and finds that asking 'why does X happen' style questions sometimes retrieves a segment that states the CLAIM (e.g. 'X happens because...') without ever retrieving the segment containing the actual REASON, even though both segments exist in the transcript.",
      "question": "Based on this page's own measured finding, what's the most likely cause?",
      "options": [
        "The query embedding is matching too literally on words like 'because,' so the retrieval system needs a cross-encoder reranker to pull the reason-containing segment up from lower in the ranked list.",
        "Whisper's own transcript segmentation can split a cause from its explanation across two adjacent segments -- retrieval correctly finds the segment containing the claim, but the specific reason is in the very next segment, which isn't guaranteed to also be retrieved.",
        "This only happens when Whisper's confidence score for a segment boundary is low, so checking confidence scores before indexing would have caught it.",
        "Sampling frames more frequently would fix this, since the missing reason must be something shown visually on the whiteboard rather than stated in the narration."
      ],
      "correct": 1,
      "explanations": [
        "A reranker can only reorder candidates that already made it into the retrieved set -- it can't recover a segment that was never pulled into the top-k in the first place. The root cause this page identifies is where Whisper drew the segmentation boundary, not literal keyword matching on 'because.'",
        "Correct. This exactly matches this page's measured 'why does the stratosphere begin to warm up' failure: the retrieved segment ended right at '...because,' and the actual reason was in the next segment, unretrieved -- a segmentation-boundary problem, not a comprehension failure.",
        "Nothing about this was a low-confidence transcription event -- Whisper split the segments at a normal, confident sentence/pause boundary. The problem is that retrieval units on the time axis, like chunks on the character axis, can separate a claim from its explanation even when each segment was transcribed perfectly.",
        "The page confirms the actual reason text is present and correctly transcribed, in the very next segment -- it isn't a visual-only fact. The fix is closing the retrieval gap on the text side (e.g. also retrieving adjacent segments), not sampling video frames more densely."
      ]
    },
    {
      "scenario": "A team's video RAG system is asked two very similar-sounding factual questions in the same short segment of video -- e.g. two different percentages for two closely related sub-topics discussed back to back. Retrieval for one question works correctly, but for the other, the top-3 results are dominated by segments near (but not exactly at) the actually-correct segment.",
      "question": "What does this page identify as the mechanism behind this kind of failure?",
      "options": [
        "Whisper must have merged the UV-C segment's text into one of the adjacent UV-B segments during transcription, so the UV-C figure no longer exists as its own separate retrievable unit.",
        "Adjacent segments discussing very similar content (e.g. two nearby percentage figures on a related topic) can score more similarly to each other by embedding similarity than the truly correct segment stands out by its specific, different content -- near-duplicate context can crowd out the exact right answer.",
        "Embedding models specifically struggle to represent numbers accurately, so any question with a percentage or count in it is inherently unreliable to retrieve with this pipeline.",
        "Retrieving the matched frame alongside the transcript segment would have resolved the ambiguity, since the whiteboard visual for each percentage looks different enough to disambiguate them."
      ],
      "correct": 1,
      "explanations": [
        "The page quotes the UV-C segment's own distinct text directly -- it exists as its own separate segment, not merged into a neighboring one. The failure is in retrieval RANKING (it wasn't pulled into the top-3), not in transcription completeness.",
        "Correct. This is exactly this page's measured UV-C finding: the right segment existed, but two adjacent segments about a very similar related topic (UV-B's percentage, generic ozone-absorption phrasing) scored highly enough to crowd it out of the top-3.",
        "The page frames this as a general near-duplicate-content problem -- three segments sharing very similar ozone-absorption phrasing -- not a specific weakness in how embedding models represent numbers. The same crowding-out effect would happen with any set of closely related, similarly-worded segments, numeric or not.",
        "Frame grounding only kicks in after a transcript segment is already retrieved -- it can't rescue a segment that never made the top-3 in the first place. The page's diagnosis for this failure is squarely about text-embedding ranking, not anything visual on the whiteboard for this question."
      ]
    },
    {
      "scenario": "Asked a question whose correct answer is a label written on a whiteboard shown in the video (and never spoken by the narrator), a transcript-only video RAG system responds that it cannot determine the answer from the given context, rather than guessing.",
      "question": "Based on this page's findings, what is the correct way to interpret this response?",
      "options": [
        "This is a failure of the system, and the fix should be lowering the retrieval similarity threshold so a plausible transcript segment gets returned even for questions the transcript doesn't actually answer.",
        "This is the CORRECT behavior for a transcript-only system: the information genuinely isn't in the text it has access to, so declining is more honest than guessing a plausible-sounding label -- the actual fix is adding frame grounding, not 'fixing' the text retrieval, which was never going to contain this answer.",
        "The system should have used a larger, more accurate Whisper model, since larger ASR models can pick up faint background text-to-speech cues that smaller models miss when text is displayed on screen.",
        "This proves transcript-only retrieval should be abandoned entirely in favor of frame-only retrieval for any video containing a whiteboard or on-screen diagram."
      ],
      "correct": 1,
      "explanations": [
        "Lowering the threshold would just force the system to return some segment regardless of relevance, trading an honest 'I don't know' for a plausible-sounding but ungrounded guess. The fix the page actually demonstrates is adding a genuinely relevant new signal (the frame), not squeezing more out of an already-exhausted one.",
        "Correct. This page states this directly: a larger or better text retrieval system was never going to find a visual-only fact in the transcript, because it was never spoken. The real fix demonstrated on this page is passing the actual frame image to a vision-capable model, not improving text retrieval further.",
        "Whisper is a speech-to-text model -- it transcribes audio only and has no mechanism for reading on-screen visual text, faint or otherwise, at any model size. Only a vision-capable step (passing the actual frame) can recover information that only ever existed visually.",
        "The page's own guidance is to add frame grounding alongside transcript retrieval, not replace it -- transcript-only retrieval scored 3/5 and works fine for narrated content. The whiteboard case shows a targeted gap that frame grounding fills, not a reason to drop transcript search altogether."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding) — a real, current example of tier 2 above: native video file input, processed as video, not decomposed into a transcript and still frames first
- [Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — an open-weight "omni" model, real current model card, that accepts text, image, audio, and video (including embedded audio) as direct input
- [VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos](https://arxiv.org/abs/2501.05874) — tier 3 above: a genuine video-native retrieval approach (video embedded directly, not frame-by-frame)
- [Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](https://arxiv.org/abs/2411.13093) — a training-free approach combining ASR transcript, OCR, and object-detection labels as auxiliary text alongside sampled frames
- [OpenAI Whisper](https://github.com/openai/whisper) — the speech-to-text model used in this recipe
- [Multimodal RAG](multimodal-rag.md) and [Vision RAG](vision-rag.md) — the CLIP-based visual embedding mechanisms this page deliberately doesn't re-implement, referenced above as what a fuller frame-search pipeline would add
