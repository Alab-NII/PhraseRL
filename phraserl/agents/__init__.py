from .template.agent import TemplateAgent
from .seq2seq.agent import Seq2SeqAgent

AGENTS = {
    "template": TemplateAgent,
    "seq2seq": Seq2SeqAgent,
}


def get_agent_cls(name):
    return AGENTS[name]
