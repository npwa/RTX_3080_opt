## AI-Assistance Disclosure

I used Claude (chat) and Claude Code throughout this project: for reading and
navigating `llama.cpp`'s CUDA source, designing and debugging kernel changes, running
and interpreting `nsys`/`ncu` profiles, and drafting parts of this write-up. I would
not have produced work at this depth, in this timeframe, without it. Tracing a
memory-access pattern through `vecdotq.cuh` and `mmvq.cu`, deriving lane-to-address
mappings, and correlating that against real `ncu` counters is the kind of work that
used to require weeks of reading design docs and internal training before adding new
production code. AI collapsed that timeline dramatically.

That doesn't make this a hands-off exercise. I've spent years working in large C
codebases, and I know exactly what's at risk when you touch a hot kernel path: subtle
correctness bugs, alignment assumptions, undefined behavior that passes on one GPU and
fails on another. Every change here went through real verification: `test-backend-ops`,
fixed-seed output comparison, and hand-checked debug output to get actual performance
numbers. When the "improved" kernel turned out to regress end-to-end throughput despite
a better proxy metric, I didn't wave that away; I profiled it further and reported it.

I see this the same way I saw static analysis tools like Klocwork earlier in my career:
a tool that makes me faster and catches more, not a replacement for engineering
judgment. Using every available tool to write, verify, and ship correct, fast code
isn't a shortcut: it's the job.
