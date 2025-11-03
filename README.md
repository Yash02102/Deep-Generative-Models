
What actually went wrong

Services were resolved from the root Lamar container (no request scope) and/or registered with the wrong lifetime (e.g., Singleton/Transient when they needed to be Scoped).

When IDisposable transients (e.g., DbContext, repositories holding buffers/streams, HttpMessageHandlers, etc.) are created from the root container, Lamar tracks them for disposal at the container level. Because the root container lives for the whole process, those objects aren’t disposed until shutdown → memory accumulates → OutOfMemoryException under load.

If any heavy dependency was accidentally Singleton (or captured by a Singleton), it pinned large graphs in memory across requests, amplifying the leak.


Why the fix worked

Moving to manual DI registration with proper Scoped lifetimes ensured each request/work item ran inside a scope that gets disposed at the end.

As a result, per-request objects are deterministically disposed, releasing memory promptly and stopping the growth that led to OOM.

