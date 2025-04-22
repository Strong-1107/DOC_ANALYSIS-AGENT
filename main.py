from datetime import datetime
import os
import time
import json
import tempfile
from typing import List, Dict
from openai import OpenAI, AssistantEventHandler
from openai.type.beta.threads.runs import ToolCallDeltaObject
from typing_extensions import override
from docx import Document



