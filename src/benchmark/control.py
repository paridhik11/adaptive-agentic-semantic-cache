"""Control-flow pauses must never be converted into provider/judge failures."""


class RunPaused(Exception):
    def __init__(self,reason,*,retry_at=None,diagnostics=None):
        super().__init__(reason)
        self.reason,self.retry_at,self.diagnostics=reason,retry_at,diagnostics
