"""Explicit bounded roles and typed read-only tool dispatch."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    argument: str
    max_length: int = 1000

    def validate(self, arguments):
        if (not isinstance(arguments,dict) or set(arguments)!={self.argument}
                or not isinstance(arguments[self.argument],str)
                or not 1<=len(arguments[self.argument])<=self.max_length):
            raise ValueError('invalid typed tool arguments')


class DocumentToolRegistry:
    VERSION='document-read-only-tools-v1'
    SPECS={'search':ToolSpec('search','query'),'read':ToolSpec('read','document_id')}

    def __init__(self, search, read):
        self._handlers={'search':search,'read':read}

    def dispatch(self, name, arguments):
        if not isinstance(name,str) or name not in self.SPECS:
            raise ValueError('tool not allowlisted')
        self.SPECS[name].validate(arguments)
        return self._handlers[name](arguments[self.SPECS[name].argument])


class DocumentationOrchestrator:
    VERSION='bounded-document-roles-v1'
    ROLES={'CACHE_LOOKUP':'cache_gateway','PLANNING':'planner',
           'TOOLS':'source_researcher','DRAFTING':'answer_writer','REVIEWING':'reviewer'}

    @classmethod
    def route(cls, state):
        if state not in cls.ROLES:raise ValueError('no specialist for state')
        return {'stage':'supervisor_route','specialist':cls.ROLES[state],
                'target_state':state,'provider_calls':0,'routing':'deterministic_state_machine'}
