# Rationale Codes

Canonical structured rationale codes used by the evaluator, prompts, and synthetic generator.

## Why Account

| Code | Description | Allowed Fact Types | Allowed Entities |
| --- | --- | --- | --- |
| `cloud_modernization_fit` | The cloud and data stack fit a modernization or migration message. | `source:crm`, `source:news` | `account` |
| `competitor_present` | Competitive footprint creates a displacement angle. | `source:crm`, `source:news`, `event:competitive_displacement` | `account`, `trigger` |
| `enterprise_icp_fit` | The account is a strong named-account enterprise ICP match. | `source:crm`, `source:news` | `account` |
| `intent_surge` | Recent high-intent web or product activity indicates active evaluation. | `source:web`, `source:product`, `event:usage_change` | `account`, `trigger` |
| `mid_market_expansion_fit` | The account fits an expansion motion even if it is not tier-one enterprise. | `source:crm`, `source:product` | `account` |
| `named_account_priority` | The account is explicitly prioritized by territory or account strategy. | `source:crm` | `account` |
| `product_usage_growth` | Product adoption or seat growth suggests expansion potential. | `source:product`, `event:usage_change` | `account`, `trigger` |
| `security_stack_match` | Observed security tooling or needs align with the offered solution. | `source:crm`, `source:contact` | `account`, `contact` |

## Why Now

| Code | Description | Allowed Fact Types | Allowed Entities |
| --- | --- | --- | --- |
| `competitive_displacement` | A live competitive event opens a replacement window. | `source:news`, `event:competitive_displacement` | `trigger`, `account` |
| `compliance_deadline` | A compliance deadline creates a time-bounded need to act. | `source:news`, `source:crm`, `event:compliance_deadline` | `trigger`, `account` |
| `expansion_recent` | Expansion signals suggest a new budget or operational moment. | `source:news`, `event:expansion` | `trigger`, `account` |
| `funding_recent` | New funding changes the urgency or affordability of a project. | `source:news`, `event:funding` | `trigger`, `account` |
| `hiring_recent` | New hiring implies active budget, project momentum, or team formation. | `source:jobs`, `event:hiring` | `trigger`, `account` |
| `leadership_change_recent` | A fresh leadership change creates a near-term buying window. | `source:news`, `event:leadership_change` | `trigger`, `account` |
| `product_launch_recent` | A recent launch or roadmap event changes the relevance of outreach. | `source:news`, `event:product_launch` | `trigger`, `account` |
| `timing_signal_recent` | A generic but fresh trigger justifies action even without a narrower why-now code. | `source:news`, `source:product`, `source:jobs` | `trigger`, `account` |
| `usage_change_recent` | Recent product-usage movement makes the timing attractive now. | `source:product`, `event:usage_change` | `trigger`, `account` |

## Why Persona

| Code | Description | Allowed Fact Types | Allowed Entities |
| --- | --- | --- | --- |
| `champion` | An internal user or operational champion is the best entry point. | `source:contact` | `contact` |
| `economic_buyer` | The main target should be an economic or budget owner. | `source:contact` | `contact` |
| `end_user_champion` | A high-engagement end user is the best near-term persona target. | `source:contact` | `contact` |
| `multithreaded_buying_center` | Multiple complementary stakeholders should be contacted in parallel. | `source:contact` | `contact` |
| `security_buyer` | Security ownership is central to deal progression. | `source:contact` | `contact` |
| `technical_buyer` | The main target should be the technical buying owner. | `source:contact` | `contact` |
| `technical_buyer_plus_security_champion` | A technical buyer and a security champion together cover the buying center. | `source:contact` | `contact` |
| `technical_champion_pair` | A technical buyer and a likely champion should be worked together. | `source:contact` | `contact` |

## Why Channel

| Code | Description | Allowed Fact Types | Allowed Entities |
| --- | --- | --- | --- |
| `email_valid` | Valid email reachability makes direct outbound appropriate. | `source:contact` | `contact` |
| `email_valid_plus_recent_web_intent` | Email is viable and recent web or product intent supports immediate outreach. | `source:web`, `source:product`, `source:contact` | `account`, `contact`, `trigger` |
| `linkedin_present` | LinkedIn is available and is the best reliable reachable channel. | `source:contact` | `contact` |
| `manual_research_required` | Existing channels are weak enough that more research is required before outreach. | `source:crm`, `source:contact` | `account`, `contact` |
| `multichannel_exec_outreach` | The account supports a coordinated multichannel outreach pattern. | `source:contact`, `source:web` | `contact`, `account` |
| `nurture_until_signal` | The account should remain in lower-touch nurture until a stronger signal appears. | `source:web`, `source:crm` | `account` |
| `phone_valid` | Phone reachability supports call-first or multichannel outreach. | `source:contact` | `contact` |
| `warm_intro_available` | Relationship evidence suggests a warm introduction channel. | `source:crm`, `source:contact` | `account`, `contact` |
