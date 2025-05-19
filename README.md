# 🧠 DeepSearch 4 All

It simply expose a OpenAI compatible LLM API, answering like a thinking LLM. But in the backend, it performs a graph-based search and introspection to answer your question.

## How do I consume it?

Well, like a standard OpenAI API. That's literally the point!

```python
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam

# ...

async with AsyncOpenAI(
    api_key="mock",
    base_url="http://localhost:8080/v1",
) as client:
    res = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            ChatCompletionUserMessageParam(
                content="What is the status of the world?",
                role="user",
            ),
        ],
    )
    print(res.choices[0].message.content)
```

## Example using OpenAI GPT 4o-mini

Qestion:

```txt
What is the status of the world?
```

Answer:

```txt
The current status of the world encompasses a complex interplay of geopolitical tensions, economic recovery challenges, environmental issues, and social dynamics, all of which can heavily influence global stability. Here is a detailed overview of each dimension:

### 1. Geopolitical Stability

#### Eastern Europe: Russia-Ukraine Conflict
- **Ongoing Conflict**: The war between Russia and Ukraine has led to significant military and humanitarian crises. As of 2023, over **7 million Ukrainians have been displaced internally**, while over **8 million have fled to neighboring countries** (UNHCR, 2023).
- **NATO’s Response**: NATO has reinforced its eastern flank, increasing troop deployments in response to Russian actions. This development is part of efforts to enhance regional security amidst heightened tensions.
- **Economic Impact**: The conflict has disrupted global energy supplies, with EU gas imports from Russia dropping by over **60%** since the war began, spurring a shift toward alternative energy sources (IEA, 2023).

#### Middle East: Israel-Hamas Conflict
- **Humanitarian Crisis**: The escalation of violence between Israel and Hamas has left over **1 million people in urgent need of assistance**, causing severe internal and external humanitarian challenges.
- **Regional Dynamics**: The ongoing conflict threatens to involve regional powers and destabilize already fragile political landscapes in neighboring countries, complicating diplomatic relations, especially for the U.S.

#### Asia: U.S.-China Relations and Taiwan
- **Military Tensions**: The Pentagon has reported that China aims to enhance its military capabilities around Taiwan by **2025**, raising alarms over potential conflicts.
- **Economic Vulnerabilities**: Conflict over Taiwan could severely disrupt global supply chains, particularly in semiconductor production, highlighting vulnerabilities in technology sectors.

### 2. Economic Conditions

- **Recovery from COVID-19**: The IMF projected a global growth rate of **5.9%** for the recovery phase in 2021, but the growth has been uneven, with developed economies recovering faster than developing ones.
- **Inflation Issues**: Inflation rates surged due to pandemic-related supply-chain disruptions, impacting many economies with rates exceeding **5%** in several regions as of late 2021 (Bartsch et al., 2021).
- **Supply Chain Disruptions**: Ongoing labor shortages and shipping constraints have resulted in product shortages, particularly in technology and consumer goods, hindering full recovery.

### 3. Environmental Challenges

- **Climate Change Effects**: The IPCC has indicated that **around 1 million species face extinction risks** due to climate stressors, with rising global temperatures linked to severe weather events and changing ecosystems.
- **Economic Costs**: By **2050**, the economic toll of climate impacts could reach **$2.5 trillion annually** due to infrastructure damage and productivity losses in sectors like agriculture.
- **Global Sustainability Efforts**: The Paris Agreement aims to limit global temperature rise to below **2°C**, calling for significant reductions in greenhouse gas emissions and investments in renewable energy. As of recent years, approximately **$2.6 trillion** has been invested globally in renewable energy from 2010 through 2020 (IRENA).

### 4. Social Issues

- **Inequalities and Unrest**: Rising inequalities in access to resources and social services are evident globally, manifesting in increased social unrest and protests in various countries.
- **Health Disparities**: The repercussions of the COVID-19 pandemic continue to highlight disparities in healthcare access, with lower-income countries facing substantial hurdles in vaccination and recovery efforts.

### Conclusion
The interconnected challenges of geopolitical tensions, economic recovery, environmental degradation, and social inequalities create a multifaceted status of the world that requires continuous monitoring and action from global leaders. Each dimension not only influences others but also highlights the urgent need for collaborative, strategic responses to promote stability, peace, and sustainable development across the globe.

**Sources**:
- United Nations High Commissioner for Refugees (UNHCR, 2023)
- International Energy Agency (IEA, 2023)
- International Monetary Fund (IMF, 2021)
- Intergovernmental Panel on Climate Change (IPCC).
- Various economic reports and analyses regarding pandemic recovery and inflation trends

(Usage: prompt=1621989, completion=57747)
```
