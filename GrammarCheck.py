import language_tool_python
from collections import Counter
tool = language_tool_python.LanguageTool('en-US')

def spell_check(tool,text):
    result_check = tool.check(text)
    error_categories = [e.category for e in result_check]
    count = Counter(error_categories)
    dict = {k: [count[k]] for k in count.keys()}
