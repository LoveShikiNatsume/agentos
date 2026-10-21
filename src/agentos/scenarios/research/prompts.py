RESEARCH_TOPIC = (
    "Compare three AI agent orchestration frameworks for production use: "
    "LangGraph, AutoGen, and CrewAI. Evaluate across dimensions: "
    "architecture design, scalability, fault tolerance, context management, "
    "observability, and community/ecosystem maturity. "
    "Target audience: engineering team deciding on a framework for "
    "building a multi-agent customer support system."
)

RESEARCH_METHODOLOGY = (
    "Research methodology: For each framework, examine official documentation, "
    "GitHub repository (stars, issues, release cadence), published benchmarks, "
    "and real-world case studies. Use structured evaluation criteria with "
    "quantitative scores (1-10) where possible."
)

PLANNER_PROMPT = (
    "You are the research planner. Create a structured research plan: "
    "define the evaluation criteria, assign weights, specify what data "
    "each researcher should collect, and define the comparison matrix format. "
    "Output: evaluation framework document."
)

PLANNER_WEIGHTS_PROMPT = (
    "Based on the evaluation criteria, assign importance weights to each "
    "dimension (architecture 20%, scalability 20%, fault tolerance 15%, "
    "context management 15%, observability 15%, ecosystem 15%). "
    "Design the scoring rubric with specific thresholds for each score level."
)

PLANNER_INSTRUCTIONS_PROMPT = (
    "Based on the evaluation framework and weights, write specific data "
    "collection instructions for each researcher: what metrics to gather, "
    "which documentation sections to review, what benchmarks to cite, "
    "and the required evidence format for each score."
)

RESEARCHER_LANGGRAPH_PROMPT = (
    "You are the LangGraph specialist researcher. Investigate LangGraph: "
    "architecture (state machine model), scalability patterns, "
    "fault handling, context/state management, built-in observability "
    "(LangSmith integration), and ecosystem maturity. "
    "Output: structured findings with evidence and scores."
)

RESEARCHER_LANGGRAPH_SCALABILITY_PROMPT = (
    "Deep-dive into LangGraph scalability: concurrent agent support via "
    "async state graph, checkpointing and state persistence mechanisms, "
    "error recovery with conditional edges, and production deployment "
    "patterns (Kubernetes, serverless)."
)

RESEARCHER_LANGGRAPH_ECOSYSTEM_PROMPT = (
    "Assess LangGraph ecosystem maturity: GitHub stars and contributor count, "
    "release cadence, LangSmith tracing integration quality, "
    "available templates and community examples, "
    "and real-world production case studies."
)

RESEARCHER_AUTOGEN_PROMPT = (
    "You are the AutoGen specialist researcher. Investigate AutoGen: "
    "architecture (conversation-based multi-agent), scalability patterns, "
    "fault handling, context management, observability capabilities, "
    "and ecosystem maturity. "
    "Output: structured findings with evidence and scores."
)

RESEARCHER_AUTOGEN_SCALABILITY_PROMPT = (
    "Deep-dive into AutoGen scalability: conversation routing overhead "
    "with many agents, multi-agent orchestration limits, "
    "state serialization for long conversations, "
    "and production deployment considerations."
)

RESEARCHER_AUTOGEN_ECOSYSTEM_PROMPT = (
    "Assess AutoGen ecosystem: Microsoft backing and support channels, "
    "GitHub activity and contributor diversity, available agent templates, "
    "published benchmarks, and enterprise adoption case studies."
)

RESEARCHER_CREWAI_PROMPT = (
    "You are the CrewAI specialist researcher. Investigate CrewAI: "
    "architecture (role-based agent crews), scalability patterns, "
    "fault handling, context/memory management, observability, "
    "and ecosystem maturity. "
    "Output: structured findings with evidence and scores."
)

RESEARCHER_CREWAI_SCALABILITY_PROMPT = (
    "Deep-dive into CrewAI scalability: crew size limits and task "
    "delegation overhead, memory and context management across agents, "
    "hierarchical process vs sequential process performance, "
    "and production readiness assessment."
)

RESEARCHER_CREWAI_ECOSYSTEM_PROMPT = (
    "Assess CrewAI ecosystem: community adoption rate, "
    "available tools and integrations (LangChain, LlamaIndex), "
    "documentation quality, commercial offerings (CrewAI Enterprise), "
    "and real-world deployment examples."
)

ANALYST_ARCH_PROMPT = (
    "You are the architecture analyst. Compare all three frameworks on "
    "architecture design: programming model, abstraction level, flexibility, "
    "extensibility, and learning curve. Produce a scored comparison."
)

ANALYST_ARCH_SCORING_PROMPT = (
    "Produce final scored ratings (1-10) for each framework on: "
    "programming model flexibility, abstraction quality, extensibility, "
    "and learning curve. Justify each score with specific evidence "
    "from the research findings."
)

ANALYST_PERF_PROMPT = (
    "You are the performance analyst. Compare all three frameworks on "
    "scalability and performance: concurrent agent support, resource efficiency, "
    "latency characteristics, and production deployment patterns."
)

ANALYST_PERF_SCORING_PROMPT = (
    "Produce final scored ratings (1-10) for each framework on: "
    "concurrent agent capacity, resource efficiency, latency characteristics, "
    "and deployment scalability. Include quantitative evidence where available."
)

ANALYST_RELIABILITY_PROMPT = (
    "You are the reliability analyst. Compare all three frameworks on "
    "fault tolerance and observability: error recovery, state persistence, "
    "monitoring/tracing, and debugging experience."
)

ANALYST_RELIABILITY_SCORING_PROMPT = (
    "Produce final scored ratings (1-10) for each framework on: "
    "error recovery robustness, state persistence quality, "
    "monitoring/tracing capability, and debugging experience. "
    "Cite specific mechanisms from research."
)

COMPARATOR_PROMPT = (
    "You are the comparison synthesizer. Given all analyst reports, "
    "build the final comparison matrix with weighted scores. "
    "Identify the winner for each dimension and overall. "
    "Highlight trade-offs and when each framework is the best choice."
)

COMPARATOR_RECOMMENDATIONS_PROMPT = (
    "Synthesize final recommendations: identify the best framework "
    "for each use case (small team, enterprise, rapid prototyping), "
    "highlight critical trade-offs, and provide a decision tree "
    "for framework selection based on team size and requirements."
)
