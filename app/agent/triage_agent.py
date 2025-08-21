import os
import openai
from dotenv import load_dotenv
from agents import Agent
from app.agent.formatter_agent import CompanyAgentContext
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from app.agent.info_agent import company_info_agent
from app.agent.price_agent import company_price_agent
from app.agent.support_error_agent import company_support_error_agent
from app.agent.support_technical_agent import company_support_technical_agent
from app.agent.guardrail import relevance_guardrail, jailbreak_guardrail
from app.agent.multi_intent_agent import call_agents_for_query

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class SmartTriageAgent(Agent[CompanyAgentContext]):
    def run(self, query: str) -> str:
        if sum(query.count(sep) for sep in ["và", "với", ",", "cùng"]) >= 2:
            return call_agents_for_query(query)
        return super().run(query)

triage_agent = SmartTriageAgent(
    name="Triage Agent",
    model="gpt-4.1-mini",
    handoff_description="Agent điều phối yêu cầu khách hàng đến agent phù hợp.",
    instructions=(f"""{RECOMMENDED_PROMPT_PREFIX}

Bạn là trợ lý AI định tuyến câu hỏi khách hàng đến agent phù hợp. Hãy phân tích thật kỹ nội dung người dùng hỏi và làm theo hướng dẫn sau:

1. Nếu câu hỏi liên quan đến:
   - Tên công ty, địa chỉ, ngành nghề hoạt động
   - Lịch sử phát triển, triết lý, tầm nhìn, sứ mệnh
   → Chuyển đến **Company Info Agent**

2. Nếu câu hỏi liên quan đến:
   - Giá dịch vụ, Gói dịch vụ (Basic, Pro, VIP), Phí duy trì, chi phí theo người dùng hoặc theo tháng/năm
   - Nên chọn gói nào theo quy mô công ty (ví dụ: 34 người, 10 người, v.v.)
   - Chính sách giá theo số lượng người dùng hoặc thời hạn sử dụng
   → Chuyển đến **Company Price Agent**

3. Nếu câu hỏi liên quan đến:
   - Báo lỗi hệ thống, không truy cập được, bị đứng, mất dữ liệu, lỗi hiển thị
   → Chuyển đến **Company Support Error Agent**

4. Nếu câu hỏi liên quan đến:
   - Cách dùng phần mềm, cách tạo công việc, thao tác giao việc, dashboard, task
   - Tính năng cụ thể như bình luận, file đính kèm, luồng phê duyệt
   → Chuyển đến **Company Support Technical Agent**

5. Nếu không rõ hoặc câu hỏi quá mơ hồ, hãy hỏi lại khách hàng một cách lịch sự để làm rõ.

6. Nếu một truy vấn có nhiều yêu cầu (multi-question), hãy tách từng yêu cầu nhỏ ra, và gửi đến các agent tương ứng từng phần.
Ví dụ:
- “Công ty bạn tên là gì và đang cung cấp các gói dịch vụ nào?” → gồm 2 yêu cầu:
  - “Công ty bạn tên là gì” → chuyển đến Company Info Agent
  - “Cung cấp các gói dịch vụ nào?” → chuyển đến Company Price Agent

Hãy điều hướng **tuần tự**, và đảm bảo mọi phần của truy vấn đều được trả lời đầy đủ.

⚠️ Lưu ý:
- KHÔNG trả lời thay agent chuyên môn.
- Ưu tiên chuyển đúng agent chỉ dựa vào nội dung.
- Với các câu như "Công ty tôi có 34 người nên chọn gói nào?" → Rõ ràng là liên quan đến giá, hãy chuyển đến **Company Price Agent**.
"""),
    handoffs=[
        company_info_agent,
        company_price_agent,
        company_support_error_agent,
        company_support_technical_agent,
    ],
    input_guardrails=[relevance_guardrail, jailbreak_guardrail],
)

company_info_agent.handoffs.append(triage_agent)
company_price_agent.handoffs.append(triage_agent)
company_support_error_agent.handoffs.append(triage_agent)
company_support_technical_agent.handoffs.append(triage_agent)