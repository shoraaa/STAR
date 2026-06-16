## Contents
- 1 Introduction
- 2 Related Works
- 3 First-Segment-Then-Aggregate (FSTA)
  - 3.1 Vehicle Routing Problems (VRPs)
  - 3.2 FSTA Decomposition
- 4 Learning to Segment (L2Seg)
  - 4.1 Neural Architecture
  - 4.2 Training
  - 4.3 Inference
- 5 Experiment
  - 5.1 L2Seg Accelerates Various Iterative Backbone Solvers
  - 5.2 L2Seg Outperforms Classic and Neural Baselines on CVRP and VRPTW
  - 5.3 Further Analysis and Discussions
- 6 Conclusion
- Reproducibility Statement
- Acknowledgment
- References
- Appendices
- Contents
- Appendix A Supplementary Definitions
  - A.1 Unstable Edges and Stable Edges
  - A.2 Capacitated Vehicle Routing Problem
  - A.3 Vehicle Routing Problem with Time Windows
- Appendix B Details of First-Segment-Then-Aggregate (FSTA)
  - B.1 More discussions on FSTA
    - B.1.1 Visualization of Unstable Edge Patterns
    - B.1.2 Visualization of Applying FSTA on
One CVRP Instance
    - B.1.3 Assumption Verification
    - B.1.4 Details of FSTA Decomposition Framework
    - B.1.5 Applying FSTA on Various VRPs
  - B.2 Proof of FSTA
    - B.2.1 CVRP
    - B.2.2 VRPTW
    - B.2.3 VRPB
    - B.2.4 1-VRPPD.
- Appendix C L2Seg Details
  - C.1 Comparative Analysis of L2Seg Against Existing Methods
  - C.2 Input Feature Design Details
  - C.3 Masking Details
  - C.4 Training Data Collection Details
  - C.5 Inference Details
- Appendix D Experimental and Implementation Details
  - D.1 Backbone solvers
  - D.2 Baselines
  - D.3 Parameters and Training Hyperparameters
  - D.4 Instance Generation
- Appendix E Additional Experiments and Analysis
  - E.1 Hyperparameter Study
  - E.2 Results on Realistic Routing Datasets
  - E.3 Results on Clustered CVRP and Heterogeneous-demand CVRP
  - E.4 Standard Deviation Comparison
  - E.5 Case Study: Comparison of Predictions of Three L2Seg Approaches
  - E.6 Unstable and Stable Edges Convergence
- Appendix F Broader Impacts
- Appendix G Large Language Models Usage

## Abstract

Abstract Iterative heuristics are widely recognized as state-of-the-art for Vehicle Routing Problems (VRPs).
In this work, we exploit a critical observation: a large portion of the solution remains stable, i.e., unchanged across search iterations , causing redundant computations, especially for large-scale VRPs with long subtours.
To address this, we pioneer the formal study of the First-Segment-Then-Aggregate (FSTA) decomposition technique to accelerate iterative solvers. FSTA preserves stable solution segments during the search, aggregates nodes within each segment into fixed hypernodes, and focuses the search only on unstable portions.
Yet, a key challenge lies in identifying which segments should be aggregated. To this end, we introduce Learning-to-Segment (L2Seg), a novel neural framework to intelligently differentiate potentially stable and unstable portions for FSTA decomposition.
We present three L2Seg variants: non-autoregressive (globally comprehensive but locally indiscriminate) , autoregressive (locally refined but globally deficient) , and their synergy.
Empirical results on CVRP and VRPTW show that L2Seg accelerates state-of-the-art solvers by 2x to 7x . We further provide in-depth analysis showing why synergy achieves the best performance. Notably, L2Seg is compatible with traditional, learning-based, and hybrid solvers, while supporting various VRPs.

## 1 Introduction

Vehicle Routing Problems (VRPs) have profound applications such as in logistics and ride-hailing, driving advances in combinatorial optimization (Laporte, 2009). As NP-hard problems, they are typically tackled with heuristics approximately. Neural Combinatorial Optimization (NCO) (Kool et al., 2018, Bengio et al., 2021, Luo et al., 2024, Berto et al., 2023) has recently introduced machine learning into VRP solving, enabling data-driven decision-making with minimal domain knowledge while matching and even surpassing the performance of meticulously designed heuristics such as Lin-Kernighan-Helsgaun (LKH)(Helsgaun, 2017) and Hybrid Genetic Search (HGS)(Vidal, 2022).

Generally, state-of-the-art VRP solvers predominantly rely on iterative search to refine solutions through local search (e.g., ruin and repair). However, as noted in Section [3](https://arxiv.org/html/2507.01037v2#S3), a significant portion of edges stabilize(^1^11Specifically, we refer stable edges as those that consistently remain in the solution across iterations, while unstable edges are likely to be re-optimized (see Appendix [A.1](https://arxiv.org/html/2507.01037v2#A1.SS1) for the formal definitions).), or their presence in the solution stops changing between iterations, as the search progresses, despite repeated local search. For example, inner edges of neighboring subtours may remain fixed while only boundary edges undergo frequent combinatorial changes. Intuitively, such stability can be inferred from customer spatial distribution and the solution properties through end-to-end learning. Yet, existing solvers overlook such opportunities, leading to redundant computations that hinder their scalability and efficiency, especially in large-scale VRPs with long subtours.

Motivated by this critical observation, we study how learning to identify such segments can accelerate iterative search solvers, a perspective yet to be explored to the best of our knowledge. To this end, we formalize a First-Segment-Then-Aggregate (FSTA) decomposition framework, which identifies stable segments in a VRP solution and then aggregates them as fixed (one or two) hypernodes with combined attributes (e.g., total demand, min/max time windows). This not only decomposes the original large problem into more tractable subproblems but also significantly accelerates the search by leveraging iterative local search to strategically focus on unstable portions. We further show that FSTA preserves solution equivalence and is broadly applicable to VRPs with diverse constraints.

To identify unstable portions for FSTA decomposition, we then introduce Learning-to-Segment (L2Seg), a novel learning-guided framework that leverages deep models to intelligently differentiate potentially stable and unstable portions, allowing dynamic decomposition for accelerated local search. Realizing this, however, is nontrivial: it involves a large combinatorial decision space requiring accurate segment grouping, and demands modeling complex interdependencies among predicted edges, constraints, spatial distribution, solution structures, and both node and edge features.

To address these challenges, L2Seg proposes encoder-decoder-styled neural models. The encoder integrates graph-level and route-level features using attention and graph neural networks, generating node embeddings that guide edge re-optimization predictions. L2Seg offers three decoders: (1) L2Seg-NAR (Non-Autoregressive): which features one-shot fast global prediction; (2) L2Seg-AR (Autoregressive), which enjoys sequential dependency modeling for high-precision local predictions; and (3) L2Seg-SYN (Synergized), which balances the strengths of both NAR and AR.
Notably, this represents a pioneering work that explores the joint decision-making between AR and NAR models in neural combinatorial optimization. Our L2Seg models are trained via a weighted cross-entropy loss on datasets labeled using a lookahead procedure: edge stability is classified based on whether its presence in the solution was changed during iterative re-optimization.

Extensive experiments on large-scale CVRPs and VRPTWs show that L2Seg accelerates backbone heuristics by 2x to 7x, enabling them to outperform state-of-the-art classic, neural, and hybrid baselines, while generalizing well across different customer distributions and problem sizes.
Notably, L2Seg exhibits strong flexibility in enhancing various solvers, including the classic LKH-3 Helsgaun (2017) solver, other orthogonal Large Neighborhood Search (LNS) methods Shaw (1998), and learning-guided decomposition method Learning-to-Delegate (L2D) Li et al. (2021). We further analyze the synergy between AR and NAR models, showing their combination achieves the best performance by integrating NAR’s global comprehension with AR’s local precision.

Our contributions are: (1) We make a critical yet underexplored insight that stable segments persist across search iterations in large-scale VRPs, causing redundant computations; (2) We formally study and theoretically prove the properties and applicabilities of First-Segment-Then-Aggregate (FSTA) for various VRPs; (3) We develop Learning-to-Segment (L2Seg), a learning-guided framework with bespoke network architecture, training, and inference for segment identification; (4) We propose autoregressive, non-autoregressive, and their synergistic deep models, pioneering the first-of-its-kind study in NCO; (5) L2Seg consistently accelerates state-of-the-art iterative VRP solvers by 2x to 7x, boosting both classic and learning-based solvers, including other decomposition frameworks.

## 2 Related Works

VRP Solvers.
Classical VRP solvers include exact methods with guarantees (Baldacci et al., 2012) and practical heuristics (Helsgaun, 2017).
Recently,
machine learning has been applied to combinatorial optimization, either end-to-end (Kwon et al., 2020, Kool et al., 2018, Fang et al., 2024, Geisler et al., 2022, Gao et al., 2024, Drakulic et al., 2023, Wang et al., 2024, Min et al., 2023, Li et al., 2023a) or learning-guided to unite data-driven insights into human solvers (Li et al., 2021, Lu et al., 2023, Huang et al., 2024; 2023, Hottung et al., 2025).
For VRPs, the former could yield competitive performance to classic methods (Luo et al., 2023, Drakulic et al., 2023), while the latter often achieve state-of-the-art performance (Zheng et al., 2024).
Among these, most effective VRP solvers rely on iterative search, including classic heuristics such as HGS (Vidal, 2022), LNS (Shaw, 1998) and LKH (Helsgaun, 2017);
neural solvers that learn local search (Ma et al., 2021; 2023, Kim et al., 2023, Hottung and Tierney, 2022, Ma et al., 2022);
neural constructive solvers integrated with search components (Hottung et al., 2022, Luo et al., 2023, Kim et al., 2021, Sun and Yang, 2023, Chalumeau et al., 2023, Kim et al., 2024, Qiu et al., 2022); and hybrid learning-guided methods like L2D (Li et al., 2021).
However, both handcrafted and neural iterative search solvers overlook the redundant computations identified in this paper, particularly in large-scale VRPs.

Decomposition for Large-scale VRPs. Scalability in VRP solvers often relies on effective decomposition that operates on solutions partially (Santini et al., 2023). This includes hand-crafted heuristics, such as LNS (Shaw, 1998) and evolutionary algorithms (Helsgaun, 2017), as well as learning-based methods such as sub-tour grouping (Zong et al., 2022), problem variant reduction (Hou et al., 2023), action space decomposition (Drakulic et al., 2023, Luo et al., 2023, Zhou et al., 2025a) and spatial-based decomposition (Zheng et al., 2024, Zhou et al., 2025b, Pan et al., 2025). In this paper, we present FSTA and L2seg, a fresh(^2^22A detailed comparison with representative decomposition methods is provided in Appendix [C.1](https://arxiv.org/html/2507.01037v2#A3.SS1)). learning-based decomposition framework that automatically detects unstable edges and aggregates stable segments. Notably, L2Seg holds potential to enhance other decomposition methods, such as LNS (Shaw, 1998) and L2D (Li et al., 2021).
While another related work  (Morabit et al., 2024) explores segment stability for re-optimization in a specific dynamic CVRP setting, our work addresses a different problem, i.e., identifying stable segments across search steps to accelerate iterative solvers. And we formally analyze the solution equivalence of FSTA across broader VRP variants. Moreover, L2Seg uniquely designs and integrates three novel deep learning models (AR, NAR, and synergized) to guide FSTA decomposition during search.

AR and NAR Models. In NCO, NAR models make global predictions like edge heatmaps (Sun and Yang, 2023, Li et al., 2023b). However, they struggle to model complex interdependencies, particularly VRP constraints. In contrast, AR models make sequential predictions, e.g., node by node selection in construction solvers (e.g., Luo et al. (2023)). AR offers stronger modeling capacity but might overlook global structure. Recent NCO works combine AR and NAR models in divide-and-conquer frameworks, with NAR for problem splitting and AR for solving (Zheng et al., 2024, Ye et al., 2024, Hou et al., 2023). We are the first to leverage their complementary strengths for joint decision-making, enabling more effective identification of unstable segments in FSTA decomposition.

## 3 First-Segment-Then-Aggregate (FSTA)

Figure: Figure 1: Percentage of re-optimized edges during iterative search using LKH-3 on 100 CVRP instances. Most edges remain unchanged, suggesting redundant calculations.
Refer to caption: https://arxiv.org/html/2507.01037/figures/redundancy/redundancy_new_1.jpg

### 3.1 Vehicle Routing Problems (VRPs)

VRPs aim to minimize total travel costs (often distance or travel time) while serving a set of customers under constraints. Formally, A VRP instance $P$ is defined on a graph $G=\left(V,E\right)$, where each node $x_{i}\in V$ represents a customer and each edge $e_{i,j}\in E$ represents traveling from $x_{i}$ to $x_{j}$ and is associated with a travel cost. For Capacitated VRP (CVRP), vehicles of capacity $C$ start and end at a depot node $x_{0}$. The sum of the demands $d_{i}$ on any route must not exceed $C$, and each customer should be served exactly once. For VRP with Time Windows (VRPTW), each customer is additionally associated with a service time $s_{i}$ and a time window $\bigl[t^{l}_{i},\,t^{r}_{i}\bigr]$ within which service must begin. See Appendix [A](https://arxiv.org/html/2507.01037v2#A1) for the formal definitions of CVRP and VRPTW.

### 3.2 FSTA Decomposition

Figure [1](https://arxiv.org/html/2507.01037v2#S3.F1) depicts that iterative search solvers perform redundant searches, reoptimizing only a small portion while many edges remain unchanged, especially in large subtours with high capacity $C$.
Inspired by Morabit et al. (2024), we formally study the decomposition technique, First-Segment-Then-Aggregate (FSTA), for accelerating iterative search solvers. As shown in the top of Figure [2](https://arxiv.org/html/2507.01037v2#S3.F2), FSTA segments the VRP solutions by identifying unstable portions, and then groups them into hypernodes with aggregated attributes. We thus expect more efficient re-optimization on the reduced problems with smaller size. More visualization of FSTA is provided in Appendix [B.1](https://arxiv.org/html/2507.01037v2#A2.SS1).

Segment Definition. Denote the solution (set of routes) of a CVRP as $\mathcal{R}=\{R^{1},R^{2},...\}$, and each route as $R^{i}=(x_{0}\rightarrow x^{i}_{1}\rightarrow x^{i}_{2}\rightarrow...\rightarrow x_{0})\in\mathcal{R}$, where the first and the last nodes in $R^{i}$ are the depot.
A segment consists of some consecutive nodes within a route. We denote the segment containing the $j^{\text{th}}$ to $k^{\text{th}}$ nodes of route $i$ as $S^{i}_{j,\,k}=(x_{j}^{i}\rightarrow...\rightarrow x_{k}^{i})$. An aggregated segment $\tilde{S}^{i}_{j,\,k}$ uses one hypernode ($\tilde{S}^{i}_{j,\,k}=\{\tilde{x}^{i}_{j,k}\}$) or two hypernodes ($\tilde{S}^{i}_{j,\,k}=\{\tilde{x}^{i}_{j},\tilde{x}^{i}_{k}\}$) with aggregated attributes (e.g. the demand of $\tilde{x}^{i}_{j,k}$ equals to $d_{j}^{i}+...+d_{k}^{i}$) to represent the non-aggregated segment $S^{i}_{j,\,k}$.

FSTA Solution Update.
After identifying unstable edges $\{e^{i}_{j_{1}},e^{i}_{j_{2}},...\}$ in each route (which will be addressed in [Section 4](https://arxiv.org/html/2507.01037v2#S4)), where each $e^{i}_{j}$ denotes the edge starting from the $j^{\text{th}}$ node in route $R^{i}$, we break these edges and group the remaining stable edges into segments.
To preserve a valid depot, edges connecting to the depot are included in the unstable edge set.
After unstable edges are removed, each route $R^{i}$ is then decomposed into multiple disjoint segments $\{x_{0},S^{i}_{1,j_{1}},S^{i}_{j_{1},j_{2}},...\}$, where $x_{0}$ is depot.
Each segment ${S}^{i}_{j,\,k}$ is then aggregated into one or two hypernodes $\tilde{S}^{i}_{j,\,k}$,
leading to a reduced problem $\tilde{P}$. We then obtain the corresponding solution $\tilde{\mathcal{R}}$ for such reduced problem, where for each $\tilde{R}^{i}\in\tilde{\mathcal{R}}$, we have $\tilde{R}^{i}=(x_{0}\rightarrow\tilde{S}^{i}_{1,j_{1}}\rightarrow\tilde{S}^{i}_{j_{1},j_{2}}...\rightarrow x_{0})$.
With fewer nodes than the original problem $P$, re-optimization with a backbone solver becomes more efficient, which is analyzed and confirmed in Appendix [B.1](https://arxiv.org/html/2507.01037v2#A2.SS1).
After re-optimization, we obtain a new solution $\tilde{\mathcal{R}}_{+}$ for the reduced problem $\tilde{P}$, which is then recovered into a solution $\mathcal{R}_{+}$ for the original problem $P$ by expanding each hypernode(s) back into its original segment of nodes. This relies on our monotonicity theorem, which guarantees that an improved solution in $\tilde{P}$ maps to an improved solution in $P$.

Theoretical Analysis. We establish a theorem proving FSTA’s feasibility and monotonicity across multiple VRP variants (e.g. CVRP, VRPTW, VRPB, and 1-VRPPD), with the proof in Appendix [B.2](https://arxiv.org/html/2507.01037v2#A2.SS2).

Theorem (Feasibility & Monotonicity). If the aggregated solution $\tilde{\mathcal{R}}_{+}$ is feasible to the aggregated problem, then $\mathcal{R}_{+}$ is also feasible to the original, non-aggregated problem. Moreover, if two feasible aggregated solutions $\tilde{\mathcal{R}}^{1}_{+}$ and $\tilde{\mathcal{R}}^{2}_{+}$ satisfy $f(\tilde{\mathcal{R}}^{1}_{+})\leq f(\tilde{\mathcal{R}}^{2}_{+})$, where $f(\cdot)$ denotes the objective function (total travel cost), their corresponding original solutions also preserve this order: $f(\mathcal{R}^{1}_{+})\leq f(\mathcal{R}^{2}_{+})$.

Figure: Figure 2: The overview of our FSTA decomposition framework (top) and the three proposed L2Seg models (bottom). L2Seg-SYN employs a four-step synergized approach: (1) problem decomposition into subproblems, (2) unstable nodes detection globally via NAR decoding, (3) clustering of NAR-predicted nodes to localize unstable regions and select initial target nodes, and (4) refining unstable edge predictions locally via AR decoding starting from these identified initial target nodes.
Refer to caption: https://arxiv.org/html/2507.01037/x1.png

## 4 Learning to Segment (L2Seg)

We introduce Learning to Segment (L2Seg), a neural framework for predicting unstable edges to guide FSTA.
We consider two paradigms: 1) Non-autoregressive (NAR) and 2) Autoregressive (AR) models. NAR models offer global predictions with an efficient single forward pass. However, they lack conditional modeling to accurately capture local dependencies.
For example, when one edge is unstable, nearby edges often show instability but not all, but NAR models may fail to distinguish them and mark all neighboring edges as unstable.
On the other hand, AR models can more natively capture local dependencies. Yet, they may miss the crucial global structure. For example, when unstable edges are distributed across distant regions, AR models may struggle to recognize and model these broader patterns.
Our approach offers three variants as shown in Figure [2](https://arxiv.org/html/2507.01037v2#S3.F2):
non-autoregressive (L2Seg-NAR), autoregressive (L2Seg-AR), and a synergized combination of both (L2Seg-SYN).

### 4.1 Neural Architecture

The autoregressive and non-autoregressive models of L2Seg share the same encoder structure. Next, we first describe the encoder, and then the two decoder architectures.

Input Feature Design.
We propose enhanced input features for L2Seg to better distinguish unstable and stable edges (see Appendix [B.1](https://arxiv.org/html/2507.01037v2#A2.SS1) for intuitions). Key features include node angularity relative to the depot and node internality, where the latter measures the proportion of nearest nodes within the same route. We consider two edge types: edges in the current solution $\mathcal{R}$ and edges connecting each node to their k-nearest neighbors. Appendix [C.2](https://arxiv.org/html/2507.01037v2#A3.SS2) provides a detailed feature description.

Encoder.
Given node features $\mathbf{X}=(\mathbf{x_{0}},\mathbf{x_{1}},\dots)$ and edge features $\mathbf{E}=\{\mathbf{e}_{0,1},\mathbf{e}_{0,2},\dots\}$, we compute the initial node embedding as $\mathbf{h}^{\mathrm{init}}_{i}\!=\!\text{Concat}(\mathbf{h}^{\mathrm{MLP}}_{i},\mathbf{h}^{\mathrm{POS}}_{i})\in\mathbb{R}^{2d_{h}}$, where $\mathbf{h}^{\mathrm{MLP}}_{i}$ and $\mathbf{h}^{\mathrm{POS}}_{i}$ are obtained by passing $\mathbf{x_{i}}$ through a multilayer perceptron (MLP) and an absolute position encoder  (Vaswani, 2017), respectively.
Next, we process the embeddings using $L_{\mathrm{TFM}}$ Transformer layers  (Vaswani, 2017) with masks to prevent computation between nodes in different routes: $\mathbf{h}^{\mathrm{TFM}}_{i}\!=\!\text{TFM}\left(\mathbf{h}^{\mathrm{init}}_{i}\right)\in\mathbb{R}^{d_{h}}$.
This step encodes local structural information from the current solution.
Finally, we compute the node embeddings $\mathbf{H}^{\mathrm{GNN}}\!=\!\{\mathbf{h}^{\mathrm{GNN}}_{i}\in\mathbb{R}^{d_{h}}\mid i=0,\dots,|V|\}$ leveraging the global graph information by using $L_{\mathrm{GNN}}$ layers of a Graph Attention Network (GAT)  (Veličković et al., 2017), where $\mathbf{H}^{\mathrm{GNN}}\!=\!\text{GNN}\left(\mathbf{H}^{\mathrm{TFM}},\mathbf{E}\right)$.

Figure: Figure 3: Architecture of L2Seg: encoder (left), NAR decoder (center), and AR decoder (right). NAR predicts unstable nodes for associated edges. AR uses a two-stage process, where the insertion bridges the deletion stage to accurately detect unstable edges locally, akin to the local search behavior.
Refer to caption: https://arxiv.org/html/2507.01037/x2.png

Non-Autoregressive Decoder. It uses an MLP with a sigmoid function to decode the probability $\mathbf{p}^{\mathrm{NAR}}$ of each node being unstable globally in one shot, so as to identify associated unstable edges:

$$ $\mathbf{p}^{\mathrm{NAR}}=\mathrm{MLP}_{\mathrm{NAR}}\left(\mathbf{H}^{\mathrm{GNN}}\right)$ (1) $$

Autoregressive Decoder.
The autoregressive decoder models unstable edge interdependence by generating them sequentially as $a=\{x_{\pi_{0}},x_{\pi_{1}},\dots\}$. Following classical local search where $k$ removed edges are reconnected via $k$ new insertions (Funke et al., 2005), the sequence alternates between deletion (identifying unstable edges) and insertion (introducing pseudo-edges that bridge to the next unstable edge), terminating at $x_{\mathrm{end}}$. Note that the “insertion" stage is designed to model dependencies between consecutive unstable edges rather than actually “insert" edges into the solution.
Formally, denote the set of edges within the current solution as $E_{\mathcal{R}}$. The decoding alternates between:
(1) Deletion ($t=2k$): Selects an unstable edge $e_{\pi_{2k},\pi_{2k+1}}\in E_{\mathcal{R}}$ based on a target node, which is either initialized at the first step (see Section [4.3](https://arxiv.org/html/2507.01037v2#S4.SS3)) or the one obtained from the previous insertion step; one of the two edges connected to this node in the current solution is then selected as unstable (more than two candidates may exist if the node is the depot);
and (2) Insertion ($t=2k+1$): Selects an new edge $e_{\pi_{2k+1},\pi_{2k+2}}\notin E_{\mathcal{R}}$ that links to the endpoint of the last unstable edge removed, exploring $O(|V|)$ potential candidates to serve as a bridge to the next unstable target node (next unstable region).
From $a$, we then identify the set of removed edges as the unstable edges, i.e., $E_{\text{unstable}}=\{e_{\pi_{0},\pi_{1}},e_{\pi_{2},\pi_{3}},\dots\}$.
Both stages employ two principal modules: Gated Recurrent Units (GRUs)  (Chung et al., 2014) to encode sequence context, and multi-head attention (MHA)  (Vaswani, 2017) for node selection. The GRU’s initial hidden state is the average of all node embeddings: $\mathbf{h}^{\text{hidden}}_{0}=\frac{1}{|V|}\sum_{i=0}^{|V|}\mathbf{h}^{\text{GNN}}_{i}$.
At step $t$, the sequence embedding is updated by
$\mathbf{h}^{\text{seq}}_{t}=\mathrm{GRU}\bigl(\mathbf{h}^{\text{hidden}}_{t-1},\,\mathbf{h}^{\text{GNN}}_{\pi_{t-1}}\bigr)$,
and the context embedding is formed by concatenating the embeddings of the initial node, the previous node, and the new sequence embedding:
$\mathbf{H}_{t}^{\text{context}}=\mathrm{Concat}\bigl(\mathbf{h}^{\text{GNN}}_{\pi_{0}},\,\mathbf{h}^{\text{GNN}}_{\pi_{t-1}},\,\mathbf{h}^{\text{seq}}_{t}\bigr)$.

Inspired by the decoder design in LEHD  (Luo et al., 2023), we use two distinct MHA modules with $L^{\mathrm{MHA}}$ layers, to decode $x_{\pi_{t}}$. Specifically, considering the size of the action space (at most 2 for deletion and $O(|V|)$ for insertion), we utilize a shallow decoder ($L_{\mathrm{delete}}^{\mathrm{MHA}}=1$) during the deletion

Figure: Figure 4: Training data construction for the AR model. Re-optimization reveals deleted edges (blue/green dashed) and inserted edges (red dashed) forming connected components (circles). For each component, depth-first search generates node sequences alternating between deletion and insertion stages, terminated by an end token as the AR model’s training label.
Refer to caption: https://arxiv.org/html/2507.01037/x3.png

stage and a deeper decoder ($L_{\mathrm{insert}}^{\mathrm{MHA}}=4$) during the insertion stage. Let $\mathbf{H}^{a}_{t}\subseteq\mathbf{H}^{\mathrm{GNN}}$ denote the set of available nodes at step $t$. During the insertion stage, we also incorporate an additional candidate
$\mathbf{h}^{\mathrm{end}}=\alpha\mathbf{h}^{\mathrm{GNN}}_{\pi_{0}}\!+\!(1-\alpha)\,\frac{1}{|V|}\sum_{i=0}^{|V|}\mathbf{h}^{\mathrm{GNN}}_{i}$, where $\alpha$ is a learnable parameter, to indicate termination of decoding, providing the AR model flexibility to determine the number of unstable edges.
Formally, the decoding at step $t$ is given as follows; note that the first 3 dimensions of $\mathbf{H}^{(L^{\mathrm{MHA}})}$ corresponds to context embeddings $\mathbf{H}_{t}^{\text{context}}$ and hence are masked from selection:

$$ $\begin{split}&\mathbf{H}^{(0)}=\mathrm{Concat}\bigl(\mathbf{H}_{t}^{\mathrm{context}},\,\mathbf{H}^{a}_{t}\bigr),\\ &\mathbf{H}^{(l)}=\mathrm{MHA}\bigl(\mathbf{H}^{(l-1)}\bigr),\\ &u_{i}=\begin{cases}(W_{q}\mathbf{h}^{\mathrm{c}})^{T}W_{k}\mathbf{h}_{i}^{(L^{\mathrm{MHA}})}\,\!\!/\!\,\sqrt{d_{h}},&\text{if }i>3,\\ -\infty,&\text{O.W.},\end{cases}\end{split}$ (2) $$

where $1\leq l\leq L^{\mathrm{MHA}}$, $W_{q}$ and $W_{k}$ are learnable matrices, and $\mathbf{h}^{\mathrm{c}}\!\in\!\mathbb{R}^{6d_{h}}$ concatenates the first three columns of $\mathbf{H}^{(0)}$ and $\mathbf{H}^{(L^{\mathrm{MHA}})}$ along the last axis. The node $x_{\pi_{t}}$ is sampled from $\mathbf{p}^{\text{AR}}_{t}=\mathrm{softmax}(\mathbf{u})$.

### 4.2 Training

We employ iterative solvers as look-ahead heuristics to detect unstable edges. We utilize imitation learning to train L2Seg models to replicate the behavior of the look-ahead heuristics.

Dataset Construction.
Let the edges in $\mathcal{R}$ be $E_{\mathcal{R}}$, and nodes indicated by edge set $E$ be $V_{E}$. Given $P$ with current solution $\mathcal{R}$, we first employ an iterative solver $\mathcal{S}$ to refine $\mathcal{R}$ and obtain $\mathcal{R}_{+}$. We then collect differing edges as $\mathcal{R}$ and $\mathcal{R}_{+}$ as $E_{\mathrm{diff}}\!=\!\bigl(E_{\mathcal{R}}\setminus E_{\mathcal{R}_{+}}\bigr)\!\cup\!\bigl(E_{\mathcal{R}_{+}}\setminus E_{\mathcal{R}}\bigr)$ (including both the deleted and newly inserted edges).
Next, we identify the set of unstable nodes $V_{\mathrm{unstable}}\!=\!V_{E_{\mathrm{diff}}}$, i.e., the set of nodes that are end points to some edge in $E_{\text{diff}}$.
We empirically observe that solution refinement typically takes place between two adjacent routes.
For the NAR model, we construct a dataset with binary labels. Each problem-label pair consists of a decomposed problem containing two adjacent routes and binary labels indicating whether each node is unstable (1) or stable (0). Formally, a node $x$ is labeled 1 if $x\in V_{\text{unstable}}$.
For the AR model,
we construct labels as node sequences preserving local dependencies among unstable edges. Nodes without local dependencies are naturally excluded through connected component partitioning. We obtain connected components $\mathcal{K}$ induced by $E_{\mathrm{diff}}$ and select those spanning at most two routes, denoted $\mathcal{K}_{\mathrm{TR}}$. For each $K\in\mathcal{K}_{\mathrm{TR}}$ containing nodes from routes $R_{i}$ and $R_{j}$, we form a subproblem $P_{K}$ with solution $\mathcal{R}_{K}=\{R_{i},R_{j}\}$. From each component $K$ (dashed circles in Figure [4](https://arxiv.org/html/2507.01037v2#S4.F4)), we extract a node sequence $y_{K}=\{x_{\pi_{0}},x_{\pi_{1}},\dots,x_{\pi_{m}},x_{\mathrm{end}}\}$ by alternating between edge deletion and insertion operations (shown in Figure [4](https://arxiv.org/html/2507.01037v2#S4.F4), second row).
These problem-label pairs $(P_{K},y_{K})$ constitute the AR model training data.

Loss Function.
To balance labels, we use weighted binary cross-entropy for the NAR model ($w_{\mathrm{pos}}>1$) and weighted cross-entropy for the AR model to balance the two stages ($w_{\mathrm{insert}}>w_{\mathrm{delete}}$).

$$ $\begin{split}L_{\mathrm{NAR}}(\mathbf{p}^{\mathrm{NAR}},y^{ij})=&-\sum_{y_{x_{k}}\in y^{ij}}w_{\mathrm{pos}}\,y_{x_{k}}\,\log\bigl(p^{\mathrm{NAR}}_{k}\bigr)+\bigl(1-y_{x_{k}}\bigr)\,\log\bigl(1-p^{\mathrm{NAR}}_{k}\bigr)\\ L_{\mathrm{AR}}(\mathbf{p}^{\mathrm{AR}},y_{K})=&-\sum_{\begin{subarray}{c}x_{\pi_{2k}}\in y_{K}\end{subarray}}w_{\mathrm{insert}}\,\log\bigl(p^{\mathrm{AR}}_{\pi_{2k}}\bigr)\,\,\,\,-\sum_{\begin{subarray}{c}x_{\pi_{2k+1}}\,\in\,y_{K}\end{subarray}}w_{\mathrm{delect}}\,\log\bigl(p^{\mathrm{AR}}_{\pi_{2k+1}}\bigr).\end{split}$ $$

### 4.3 Inference

We describe the synergized inference that combines the benefits of global structural awareness from NAR with the local precision from AR, followed by two variants using only NAR or AR.

Synergized Prediction (L2Seg-SYN).
L2Seg-SYN’s inference pipeline for detecting unstable edges consists of four steps: (1) problem decomposition, (2) global unstable node detection via NAR decoding, (3) representative initial node identification for AR decoding based on NAR predictions, and (4) local unstable edge detection using AR decoding.

Given a problem $P$ with solution $\mathcal{R}$, we partition $P$ into approximately $|\mathcal{R}|$ subproblems, $\mathcal{P}_{\mathrm{TR}}$, by grouping nodes from all two adjacent sub-tour pairs. For each subproblem in $\mathcal{P}_{\mathrm{TR}}$, the NAR model predicts unstable nodes as $\hat{y}_{\mathrm{NAR}}=\{x_{i}\mid p^{\mathrm{NAR}}_{i}\geq\eta\}$, where $\eta$ is a predefined threshold.
We then refine unstable edge detection with the AR model within regions identified by the NAR prediction. To reduce redundant decoding efforts on neighboring unstable nodes, we first group unstable nodes into $n_{\mathrm{KMEANS}}$ clusters using the $K$-means algorithm, and select the node with the highest $p_{i}^{\mathrm{NAR}}$ within each cluster as the starting point for AR decoding.
The AR model then detects unstable edges based on these initial nodes. Finally, we aggregate unstable edges from all subproblems in $\mathcal{P}_{\mathrm{TR}}$ as the final unstable edge set for $P$ given the current solution $\mathcal{R}$.

Non-Autoregressive Prediction (L2Seg-NAR).
L2Seg-NAR uses only the NAR model for predictions. It identifies unstable nodes and marks all connected edges as unstable.

Autoregressive Prediction (L2Seg-AR).
L2Seg-AR exclusively uses the AR model. Instead of using the NAR model, it assumes all nodes may be unstable, applying the $K$-means algorithm on all nodes. It then selects the node closest to each cluster center as the initial node for AR-based decoding.

## 5 Experiment

Our decomposition-based FSTA and L2Seg excel on large-scale problems. In this section, we first evaluate how L2Seg‑AR, L2Seg‑NAR, and L2Seg‑SYN accelerate various learning and non-learning iterative solvers on large‑capacity CVRPs with long subtours. Next, we compare L2Seg against state-of-the-art baselines on standard benchmark CVRP and VRPTW instances. Finally, we provide in‑depth analyses of our pipeline. Additional results on CVRPLib benchmarks, clustered CVRP, heterogeneous-demand CVRP, a case study, and further discussions are presented in Appendix [E](https://arxiv.org/html/2507.01037v2#A5).

Figure: Table 1: Performance comparisons of our proposed L2Seg-NAR, L2Seg-AR, and L2Seg-SYN when accelerating three backbone solvers, LKH-3, LNS, and L2D, on the large‑capacity CVRP instances. We report the objective value, improvement gain (%), and the time. The gains (the higher the better) are w.r.t. the performance of each backbone solver. Time limits were set to be 150s for CVRP2k and 240s for CVRP5k, respectively.
Refer to caption: https://arxiv.org/html/2507.01037/figures/lc_cvrp2kc500.jpg

Backbone Solvers. We apply L2Seg to three representative backbones: LKH-3 (Helsgaun, 2017) (classic heuristic), LNS (Shaw, 1998) (decomposition framework), and L2D (Li et al., 2021) (learning-guided hybrid solvers) to demonstrates the broad applicability. See Appendix [D.1](https://arxiv.org/html/2507.01037v2#A4.SS1) for details.

Baselines. We include state-of-the-art classic solvers (LKH-3 (Helsgaun, 2017), HGS (Vidal, 2022)), neural solvers (BQ (Drakulic et al., 2023), LEHD (Luo et al., 2023), ELG (Gao et al., 2024), ICAM (Zhou et al., 2024), L2R (Zhou et al., 2025a),
SIL (Luo et al., 2024)), and learning-based divide-and-conquer methods (GLOP (Ye et al., 2024), TAM (Hou et al., 2023), UDC (Zheng et al., 2024), L2D (Li et al., 2021), NDS (Hottung et al., 2025)). We rerun LKH-3, LNS, L2D, and NDS and report results from Zheng et al. (2024), Luo et al. (2024) for other baselines using the same benchmarks. See Appendix [D.2](https://arxiv.org/html/2507.01037v2#A4.SS2) for baseline setup details and Appendix [D.3](https://arxiv.org/html/2507.01037v2#A4.SS3) for L2Seg hyperparameters.

Data Distribution. We generate all training and test instances following prior works Zheng et al. (2024) for CVRP and Solomon (1987) for VRPTW. See Appendix [D.4](https://arxiv.org/html/2507.01037v2#A4.SS4) for details. For Section [5.1](https://arxiv.org/html/2507.01037v2#S5.SS1), results are averaged over 100 large‑scale CVRP test instances at 2k and 5k scales (capacities 500 and 1,000, respectively). For Section [5.2](https://arxiv.org/html/2507.01037v2#S5.SS2), we follow standard NCO benchmarks, reporting averaged results on 1k, 2k, and 5k test datasets with 1,000 CVRP and 100 VRPTW instances per scale.

Evaluation and Metric.
We impose time limits of 150s, 240s, and 300s for CVRP1k, 2k, and 5k, and 120s, 240s, and 600s for VRPTW1k, 2k and 5k, where each solver may finish a few seconds ($<$ 10s) beyond its limit. We set $\eta=0.6$ and $n_{\text{KMEANS}}=3$ for our L2Seg. We report averaged cost and per-instance solve time for all cases, and report percentage improvements over backbone in Section [5.1](https://arxiv.org/html/2507.01037v2#S5.SS1) and gaps to HGS (the best heuristic solvers) for both CVRP and VRPTW in Section [5.2](https://arxiv.org/html/2507.01037v2#S5.SS2).

### 5.1 L2Seg Accelerates Various Iterative Backbone Solvers

We first verify the effectiveness of the three L2Seg variants to enhance backbone solvers.
Table [1](https://arxiv.org/html/2507.01037v2#S5.T1) presents results on large‑capacity, uniformly distributed CVRPs with long subtours. All L2Seg variants consistently improve each backbone across all problem scales. Also, performance gains are larger for weaker backbones. While L2Seg‑AR and L2Seg‑NAR each boost performance, their combination (L2Seg‑SYN) delivers the best solutions. Figure [5](https://arxiv.org/html/2507.01037v2#S5.F5) plots average objective curves over time, which reveal 2x to 7x speedups on the backbone solvers with L2Seg‑SYN. Remarkably, L2Seg‑augmentation lets weaker solvers surpass stronger ones (e.g., LKH‑3 + L2Seg‑SYN outperforms vanilla LNS).

**Table 2: Performance comparisons of our L2Seg-SYN-L2D against baselines on benchmark CVRP and VRPTW instances. The gap % (lower the better) is w.r.t. the performance of HGS.**
| Methods | CVRP1k | CVRP2k | CVRP5k |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Obj.$\downarrow$ | Gap$\downarrow$ | Time$\downarrow$ | Obj.$\downarrow$ | Gap$\downarrow$ | Time$\downarrow$ | Obj.$\downarrow$ | Gap$\downarrow$ | Time$\downarrow$ |
| HGS (Vidal, 2022) | 41.20 | 0.00% | 5m | 57.20 | 0.00% | 5m | 126.20 | 0.00% | 5m |
| LKH-3 (Helsgaun, 2017) | 42.98 | 4.32% | 6.6m | 57.94 | 1.29% | 11.4m | 175.70 | 39.22% | 2.5m |
| LNS (Shaw, 1998) | 42.44 | 3.01% | 2.5m | 57.62 | 0.73% | 4.0m | 126.58 | 0.30% | 5.0m |
| BQ (Drakulic et al., 2023) | 44.17 | 7.21% | 55s | 62.59 | 9.42% | 3m | 139.80 | 10.78% | 45m |
| LEHD (Luo et al., 2023) | 43.96 | 6.70% | 1.3m | 61.58 | 7.66% | 9.5m | 138.20 | 9.51% | 3h |
| ELG (Gao et al., 2024) | 43.58 | 5.78% | 15.6m | - | - | - | - | - | - |
| ICAM (Zhou et al., 2024) | 43.07 | 4.54% | 26s | 61.34 | 7.24% | 3.7m | 136.90 | 8.48% | 50m |
| L2R (Zhou et al., 2025a) | 44.20 | 7.28% | 34.2s | - | - | - | 131.10 | 3.88% | 1.8m |
| SIL (Luo et al., 2024) | 42.00 | 1.94% | 1.3m | 57.10 | -0.17% | 2.4m | 123.10 | -2.52% | 5.9m |
| TAM(LKH-3) (Hou et al., 2023) | 46.30 | 12.38% | 4m | 64.80 | 13.29% | 9.6m | 144.60 | 14.58% | 35m |
| GLOP-G(LKH-3) (Ye et al., 2024) | 45.90 | 11.41% | 2m | 63.02 | 10.52% | 2.5m | 140.40 | 11.25% | 8m |
| UDC (Zheng et al., 2024) | 43.00 | 4.37% | 1.2h | 60.01 | 4.9% | 2.15h | 136.70 | 8.32% | 16m |
| L2D (Li et al., 2021) | 42.07 | 2.11% | 2.5m | 57.44 | 0.42% | 4.2m | 126.48 | 0.22% | 5.3m |
| NDS (Hottung et al., 2025) | 41.16 | -0.01% | 2.5m | 56.11 | -1.91% | 4m | - | - | - |
| L2Seg-SYN-LKH-3 | 41.42 | 0.53% | 2.5m | 56.37 | -1.45% | 4.4m | 122.34 | -3.16% | 5.1m |
| L2Seg-SYN-LNS | 41.36 | 0.39% | 2.5m | 56.08 | -1.96% | 4.1ms | 121.96 | -3.48% | 5.1m |
| L2Seg-SYN-L2D | 41.23 | 0.07% | 2.5m | 56.05 | -2.01% | 4.1m | 121.87 | -3.55% | 5.1m |
| Methods | VRPTW1k | VRPTW2k | VRPTW5k |  |  |  |  |  |  |
|  | Obj.$\downarrow$ | Gap$\downarrow$ | Time$\downarrow$ | Obj. $\downarrow$ | Gap$\downarrow$ | Time$\downarrow$ | Obj.$\downarrow$ | Gap $\downarrow$ | Time$\downarrow$ |
| HGS (Vidal, 2022) | 90.35 | 0.00% | 2m | 173.46 | 0.00% | 4m | 344.2 | 0.00% | 10m |
| LKH-3 (Helsgaun, 2017) | 91.32 | 1.07% | 2m | 174.25 | 0.46% | 4m | 353.2 | 2.61% | 10m |
| LNS (Shaw, 1998) | 88.12 | -2.47% | 2m | 165.42 | -4.64% | 4m | 338.5 | -1.66% | 10m |
| L2D (Li et al., 2021) | 88.01 | -2.59% | 2m | 164.12 | -5.38% | 4m | 335.2 | -2.61% | 10m |
| NDS (Hottung et al., 2025) | 87.54 | -3.11% | 2m | 167.48 | -3.45% | 4m | - | - | - |
| L2Seg-SYN-LKH-3 | 88.65 | -1.88% | 2m | 169.24 | -2.43% | 4m | 345.2 | 0.29% | 10m |
| L2Seg-SYN-LNS | 87.31 | -3.36% | 2m | 163.94 | -5.49% | 4m | 334.1 | -2.93% | 10m |
| L2Seg-SYN-L2D | 87.25 | -3.43% | 2m | 163.74 | -5.60% | 4m | 333.4 | -3.14% | 10m |

### 5.2 L2Seg Outperforms Classic and Neural Baselines on CVRP and VRPTW

We evaluate the highest-performing L2Seg-SYN implementation with three distinct backbone solvers and compare against state-of-the-art classical and neural approaches.
As demonstrated in Table [2](https://arxiv.org/html/2507.01037v2#S5.T2), L2Seg surpasses both classical and neural baselines on CVRP and VRPTW benchmarks.
For CVRP, L2Seg achieves superior performance within comparable computational time relative to competitive classical solvers, including HGS on larger problem instances. It also outperforms the state-of-the-art learning-based constructive solver SIL (Luo et al., 2024) and divide-and-conquer solver L2D (Li et al., 2021) across all problem scales.
For VRPTW, L2Seg exceeds all classical and learning-based solvers across various scales under identical time constraints, with performance advantages increasing as problem size grows.
Notably, L2Seg consistently enhances performance when integrated with any backbone solver, demonstrating its versatility. Additional analyses are provided in Appendix [E](https://arxiv.org/html/2507.01037v2#A5).

**Table 3: Performance of L2Seg‑SYN v.s. Random FSTA to accelerate LNS on CVRP instances.**
| Methods | LNS (Backbone) | Random FSTA (40%) | Random FSTA (60%) | L2Seg-SYN w/o Enhanced Features | L2Seg-SYN |
| --- | --- | --- | --- | --- | --- |
| CVRP2k | 44.92 | 46.24 | 46.89 | 43.65 | 43.42 |
| CVRP5k | 64.69 | 66.72 | 65.92 | 64.22 | 63.94 |

Figure: Table 4: Model prediction analysis of L2Seg-LNS on CVRP2k.
Refer to caption: https://arxiv.org/html/2507.01037/x4.png

### 5.3 Further Analysis and Discussions

Ablation Study. Table [3](https://arxiv.org/html/2507.01037v2#S5.T3) compares the LNS backbone; random FSTA with 40% and 60% of edges arbitrarily marked as unstable; L2Seg‑SYN w/o enhanced features; and full L2Seg‑SYN. Results show that Random FSTA worsens performance; and only full L2Seg‑SYN with enhanced features achieves top performance. This confirms that L2Seg’s learnable, feature‑guided segmentation is indispensable for preserving high‑quality segments in FSTA for boosting backbone solvers.

Figure: Figure 7: Statistic values of Size (reduced/original ratio), Recall, and TNR across three L2Seg-SYN configurations.
Refer to caption: https://arxiv.org/html/2507.01037/figures/local_opt.png

High Recall or High TNR?
Higher Recall allows more unstable edges to be reoptimized, potentially improving performance, while higher TNR reduces problem size and runtime.
However, due to learning imprecision, pursuing high TNRs often reduces Recall, causing premature convergence. Figure [7](https://arxiv.org/html/2507.01037v2#S5.F7) shows that for L2Seg-SYN, fixing too few (left: high Recall, low TNR) or too many (right: high TNR, low Recall) degrades performance. Ours (middle) balances this tradeoff for optimal performance.

Why NAR+AR Is the Best? Figure [6](https://arxiv.org/html/2507.01037v2#S5.F6) shows a conceptual illustration of the model’s behaviour across L2Seg variants (See Appendix [E.5](https://arxiv.org/html/2507.01037v2#A5.SS5) for a real case-study). L2Seg-NAR identifies unstable regions but over-classifies due to the lack of dependency modeling, while L2Seg-AR models dependencies but struggles with initial detection. L2Seg-SYN achieves the complementary synergy. Moreover, Table [4](https://arxiv.org/html/2507.01037v2#S5.T4) further shows that L2Seg-SYN achieves the best balanced Recall and TNR for the best performance.

## 6 Conclusion

This work introduces Learning-to-Segment (L2Seg), a novel learning-guided framework that accelerates state-of-the-art iterative solvers for large-scale VRPs by 2x to 7x. We formalize the FSTA decomposition and employ a specialized encoder-decoder architecture to dynamically differentiate potentially unstable and stable segments in FSTA. L2Seg features three variants, L2Seg-NAR, L2Seg-AR, and L2Seg-SYN, pioneering the synergy of AR and NAR models in NCO. Extensive results demonstrate L2Seg’s state-of-the-art performance on representative CVRP and VRPTW and flexibility in boosting classic and learning-based solvers, including other decomposition frameworks. One potential limitation is that L2Seg is not guaranteed to boost all VRP solvers across all VRP variants. Future work includes: (1) extending L2Seg to accelerate additional VRP solvers (e.g., Vidal (2022)); (2) applying L2Seg to more VRP variants and other combinatorial optimization problems; and (3) expanding the synergy between AR and NAR models to the broader NCO community.

## Reproducibility Statement

We provide comprehensive technical details in the appendices: architecture and input features (Appendix [D.3](https://arxiv.org/html/2507.01037v2#A4.SS3)), data generation (Appendix [D.4](https://arxiv.org/html/2507.01037v2#A4.SS4)), training procedures (Appendix [C.4](https://arxiv.org/html/2507.01037v2#A3.SS4)), and experimental setup (Section [5](https://arxiv.org/html/2507.01037v2#S5)). The complete codebase, including code and pre-trained models, will be released on GitHub under the MIT License upon publication.

## Acknowledgment

This research was supported by a gift from Mathworks, as well as partial support from the MIT Amazon Science Hub, the National Science Foundation (NSF) award 2149548 and CAREER award 2239566, and an Amazon Robotics Fellowship. The authors acknowledge the MIT SuperCloud and Lincoln Laboratory Supercomputing Center for providing the high-performance computing resources that contributed to the research results reported in this paper.
We particularly thank Andrea Lodi for the insightful discussions throughout this project, especially during its initial stages. We also thank Zhongxia "Zee" Yan, Jianan Zhou, and Samitha Samaranayake for their valuable input.

## References

- Laporte (2009)
Gilbert Laporte.
Fifty years of vehicle routing.
*Transportation science*, 43(4):408–416,
2009.
- Kool et al. (2018)
Wouter Kool, Herke van Hoof, and Max Welling.
Attention, learn to solve routing problems!
In *International Conference on Learning Representations*, 2018.
- Bengio et al. (2021)
Yoshua Bengio, Andrea Lodi, and Antoine Prouvost.
Machine learning for combinatorial optimization: a methodological
tour d’horizon.
*European Journal of Operational Research*, 290(2):405–421, 2021.
- Luo et al. (2024)
Fu Luo, Xi Lin, Zhenkun Wang, Xialiang Tong, Mingxuan Yuan, and Qingfu Zhang.
Self-improved learning for scalable neural combinatorial
optimization.
*arXiv preprint arXiv:2403.19561*, 2024.
- Berto et al. (2023)
Federico Berto, Chuanbo Hua, Junyoung Park, Laurin Luttmann, Yining Ma, Fanchen
Bu, Jiarui Wang, Haoran Ye, Minsu Kim, Sanghyeok Choi, Nayeli Gast Zepeda,
André Hottung, Jianan Zhou, Jieyi Bi, Yu Hu, Fei Liu, Hyeonah Kim, Jiwoo
Son, Haeyeon Kim, Davide Angioni, Wouter Kool, Zhiguang Cao, Jie Zhang,
Kijung Shin, Cathy Wu, Sungsoo Ahn, Guojie Song, Changhyun Kwon, Lin Xie, and
Jinkyoo Park.
RL4CO: an extensive reinforcement learning for combinatorial
optimization benchmark.
*arXiv preprint arXiv:2306.17100*, 2023.
- Helsgaun (2017)
Keld Helsgaun.
An extension of the lin-kernighan-helsgaun tsp solver for constrained
traveling salesman and vehicle routing problems.
*Roskilde: Roskilde University*, 12:966–980, 2017.
- Vidal (2022)
Thibaut Vidal.
Hybrid genetic search for the cvrp: Open-source implementation and
swap* neighborhood.
*Computers & Operations Research*, 140:105643, 2022.
- Shaw (1998)
Paul Shaw.
Using constraint programming and local search methods to solve
vehicle routing problems.
In *International conference on principles and practice of
constraint programming*, pages 417–431. Springer, 1998.
- Li et al. (2021)
Sirui Li, Zhongxia Yan, and Cathy Wu.
Learning to delegate for large-scale vehicle routing.
*Advances in Neural Information Processing Systems*,
34:26198–26211, 2021.
- Baldacci et al. (2012)
Roberto Baldacci, Aristide Mingozzi, and Roberto Roberti.
Recent exact algorithms for solving the vehicle routing problem under
capacity and time window constraints.
*European Journal of Operational Research*, 218(1):1–6, 2012.
- Kwon et al. (2020)
Yeong-Dae Kwon, Jinho Choo, Byoungjip Kim, Iljoo Yoon, Youngjune Gwon, and
Seungjai Min.
POMO: Policy optimization with multiple optima for
reinforcement learning.
In *Advances in Neural Information Processing Systems*,
volume 33, pages 21188–21198, 2020.
- Fang et al. (2024)
Han Fang, Zhihao Song, Paul Weng, and Yutong Ban.
Invit: A generalizable routing problem solver with invariant nested
view transformer.
In *Forty-first International Conference on Machine Learning*,
2024.
- Geisler et al. (2022)
Simon Geisler, Johanna Sommer, Jan Schuchardt, Aleksandar Bojchevski, and
Stephan Günnemann.
Generalization of neural combinatorial solvers through the lens of
adversarial robustness.
In *International Conference on Learning Representations*, 2022.
- Gao et al. (2024)
Chengrui Gao, Haopu Shang, Ke Xue, Dong Li, and Chao Qian.
Towards generalizable neural solvers for vehicle routing problems via
ensemble with transferrable local policy.
In *Proceedings of the Thirty-First International Joint
Conference on Artificial Intelligence*, 2024.
- Drakulic et al. (2023)
Darko Drakulic, Sofia Michel, Florian Mai, Arnaud Sors, and Jean-Marc Andreoli.
BQ-NCO: Bisimulation quotienting for generalizable neural
combinatorial optimization.
In *Advances in Neural Information Processing Systems*, 2023.
- Wang et al. (2024)
Chaoyang Wang, Pengzhi Cheng, Jingze Li, and Weiwei Sun.
Leader reward for pomo-based neural combinatorial optimization.
*arXiv preprint arXiv:2405.13947*, 2024.
- Min et al. (2023)
Yimeng Min, Yiwei Bai, and Carla P Gomes.
Unsupervised learning for solving the travelling salesman problem.
*Advances in Neural Information Processing Systems*, 2023.
- Li et al. (2023a)
Yang Li, Jinpei Guo, Runzhong Wang, and Junchi Yan.
From distribution learning in training to gradient search in testing
for combinatorial optimization.
*Advances in Neural Information Processing Systems*,
2023a.
- Lu et al. (2023)
Han Lu, Zenan Li, Runzhong Wang, Qibing Ren, Xijun Li, Mingxuan Yuan, Jia Zeng,
Xiaokang Yang, and Junchi Yan.
ROCO: A general framework for evaluating robustness of
combinatorial optimization solvers on graphs.
In *International Conference on Learning Representations*, 2023.
- Huang et al. (2024)
Taoan Huang, Aaron M Ferber, Arman Zharmagambetov, Yuandong Tian, and Bistra
Dilkina.
Contrastive predict-and-search for mixed integer linear programs.
In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian
Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors,
*Proceedings of the 41st International Conference on Machine Learning*,
volume 235 of *Proceedings of Machine Learning Research*, pages
19757–19771. PMLR, 21–27 Jul 2024.
URL [https://proceedings.mlr.press/v235/huang24f.html](https://proceedings.mlr.press/v235/huang24f.html).
- Huang et al. (2023)
Taoan Huang, Aaron M Ferber, Yuandong Tian, Bistra Dilkina, and Benoit Steiner.
Searching large neighborhoods for integer linear programs with
contrastive learning.
In *International Conference on Machine Learning*, pages
13869–13890. PMLR, 2023.
- Hottung et al. (2025)
André Hottung, Paula Wong-Chung, and Kevin Tierney.
Neural deconstruction search for vehicle routing problems.
*Transactions on Machine Learning Research*, 2025.
- Luo et al. (2023)
Fu Luo, Xi Lin, Fei Liu, Qingfu Zhang, and Zhenkun Wang.
Neural combinatorial optimization with heavy decoder: Toward large
scale generalization.
*Advances in Neural Information Processing Systems*,
36:8845–8864, 2023.
- Zheng et al. (2024)
Zhi Zheng, Changliang Zhou, Tong Xialiang, Mingxuan Yuan, and Zhenkun Wang.
Udc: A unified neural divide-and-conquer framework for large-scale
combinatorial optimization problems.
In *Advances in Neural Information Processing Systems*, 2024.
- Ma et al. (2021)
Yining Ma, Jingwen Li, Zhiguang Cao, Wen Song, Le Zhang, Zhenghua Chen, and
Jing Tang.
Learning to iteratively solve routing problems with dual-aspect
collaborative transformer.
In *Advances in Neural Information Processing Systems*,
volume 34, pages 11096–11107, 2021.
- Ma et al. (2023)
Yining Ma, Zhiguang Cao, and Yeow Meng Chee.
Learning to search feasible and infeasible regions of routing
problems with flexible neural k-opt.
In *Advances in Neural Information Processing Systems*,
volume 36, 2023.
- Kim et al. (2023)
Minjun Kim, Junyoung Park, and Jinkyoo Park.
Learning to cross exchange to solve min-max vehicle routing problems.
In *The Eleventh International Conference on Learning
Representations*, 2023.
- Hottung and Tierney (2022)
André Hottung and Kevin Tierney.
Neural large neighborhood search for routing problems.
*Artificial Intelligence*, page 103786, 2022.
- Ma et al. (2022)
Yining Ma, Jingwen Li, Zhiguang Cao, Wen Song, Hongliang Guo, Yuejiao Gong, and
Yeow Meng Chee.
Efficient neural neighborhood search for pickup and delivery
problems.
In *Proceedings of the Thirty-First International Joint
Conference on Artificial Intelligence, IJCAI-22*, pages 4776–4784, 7 2022.
- Hottung et al. (2022)
André Hottung, Yeong-Dae Kwon, and Kevin Tierney.
Efficient active search for combinatorial optimization problems.
In *International Conference on Learning Representations*, 2022.
- Kim et al. (2021)
Minsu Kim, Jinkyoo Park, and joungho kim.
Learning collaborative policies to solve np-hard routing problems.
In *Advances in Neural Information Processing Systems*,
volume 34, pages 10418–10430, 2021.
- Sun and Yang (2023)
Zhiqing Sun and Yiming Yang.
Difusco: Graph-based diffusion solvers for combinatorial
optimization.
In *Advances in Neural Information Processing Systems*, 2023.
- Chalumeau et al. (2023)
Felix Chalumeau, Shikha Surana, Clément Bonnet, Nathan Grinsztajn, Arnu
Pretorius, Alexandre Laterre, and Thomas D Barrett.
Combinatorial optimization with policy adaptation using latent space
search.
In *Advances in Neural Information Processing Systems*, 2023.
- Kim et al. (2024)
Minsu Kim, Sanghyeok Choi, Jiwoo Son, Hyeonah Kim, Jinkyoo Park, and Yoshua
Bengio.
Ant colony sampling with gflownets for combinatorial optimization.
*arXiv preprint arXiv:2403.07041*, 2024.
- Qiu et al. (2022)
Ruizhong Qiu, Zhiqing Sun, and Yiming Yang.
Dimes: A differentiable meta solver for combinatorial optimization
problems.
*Advances in Neural Information Processing Systems*,
35:25531–25546, 2022.
- Santini et al. (2023)
Alberto Santini, Michael Schneider, Thibaut Vidal, and Daniele Vigo.
Decomposition strategies for vehicle routing heuristics.
*INFORMS Journal on Computing*, 35(3):543–559, 2023.
- Zong et al. (2022)
Zefang Zong, Hansen Wang, Jingwei Wang, Meng Zheng, and Yong Li.
Rbg: Hierarchically solving large-scale routing problems in logistic
systems via reinforcement learning.
In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge
Discovery and Data Mining*, pages 4648–4658, 2022.
- Hou et al. (2023)
Qingchun Hou, Jingwei Yang, Yiqiang Su, Xiaoqing Wang, and Yuming Deng.
Generalize learned heuristics to solve large-scale vehicle routing
problems in real-time.
In *The Eleventh International Conference on Learning
Representations*, 2023.
- Zhou et al. (2025a)
Changliang Zhou, Xi Lin, Zhenkun Wang, and Qingfu Zhang.
L2r: Learning to reduce search space for generalizable neural routing
solver.
*arXiv preprint arXiv:2503.03137*, 2025a.
- Zhou et al. (2025b)
Shipei Zhou, Yuandong Ding, Chi Zhang, Zhiguang Cao, and Yan Jin.
Dualopt: A dual divide-and-optimize algorithm for the large-scale
traveling salesman problem.
*arXiv preprint arXiv:2501.08565*, 2025b.
- Pan et al. (2025)
Yuxin Pan, Ruohong Liu, Yize Chen, Zhiguang Cao, and Fangzhen Lin.
Hierarchical learning-based graph partition for large-scale vehicle
routing problems.
*arXiv preprint arXiv:2502.08340*, 2025.
- Morabit et al. (2024)
Mouad Morabit, Guy Desaulniers, and Andrea Lodi.
Learning to repeatedly solve routing problems.
*Networks*, 83(3):503–526, 2024.
- Li et al. (2023b)
Yang Li, Jinpei Guo, Runzhong Wang, and Junchi Yan.
T2t: From distribution learning in training to gradient search in
testing for combinatorial optimization.
In *Advances in Neural Information Processing Systems*,
2023b.
- Ye et al. (2024)
Haoran Ye, Jiarui Wang, Helan Liang, Zhiguang Cao, Yong Li, and Fanzhang Li.
Glop: Learning global partition and local construction for solving
large-scale routing problems in real-time.
In *Proceedings of the AAAI Conference on Artificial
Intelligence*, 2024.
- Vaswani (2017)
A Vaswani.
Attention is all you need.
*Advances in Neural Information Processing Systems*, 2017.
- Veličković et al. (2017)
Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero,
Pietro Lio, and Yoshua Bengio.
Graph attention networks.
*arXiv preprint arXiv:1710.10903*, 2017.
- Funke et al. (2005)
Birger Funke, Tore Grünert, and Stefan Irnich.
Local search for vehicle routing and scheduling problems: Review and
conceptual integration.
*Journal of heuristics*, 11:267–306, 2005.
- Chung et al. (2014)
Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio.
Empirical evaluation of gated recurrent neural networks on sequence
modeling.
*arXiv preprint arXiv:1412.3555*, 2014.
- Zhou et al. (2024)
Changliang Zhou, Xi Lin, Zhenkun Wang, Xialiang Tong, Mingxuan Yuan, and Qingfu
Zhang.
Instance-conditioned adaptation for large-scale generalization of
neural combinatorial optimization.
*arXiv preprint arXiv:2405.01906*, 2024.
- Solomon (1987)
Marius M Solomon.
Algorithms for the vehicle routing and scheduling problems with time
window constraints.
*Operations research*, 35(2):254–265, 1987.
- Goetschalckx and Jacobs-Blecha (1989)
Marc Goetschalckx and Charlotte Jacobs-Blecha.
The vehicle routing problem with backhauls.
*European Journal of Operational Research*, 42(1):39–51, 1989.
- Martinovic et al. (2008)
Goran Martinovic, Ivan Aleksi, and Alfonzo Baumgartner.
Single-commodity vehicle routing problem with pickup and delivery
service.
*Mathematical Problems in Engineering*, 2008(1):697981, 2008.
- Uchoa et al. (2017)
Eduardo Uchoa, Diego Pecin, Artur Pessoa, Marcus Poggi, Thibaut Vidal, and
Anand Subramanian.
New benchmark instances for the capacitated vehicle routing problem.
*European Journal of Operational Research*, 257(3):845–858, 2017.
- Arnold et al. (2019)
Florian Arnold, Michel Gendreau, and Kenneth Sörensen.
Efficiently solving very large-scale routing problems.
*Computers & operations research*, 107:32–42, 2019.

## Appendices

## Contents

## Appendix A Supplementary Definitions

### A.1 Unstable Edges and Stable Edges

Unstable edges refer to edges that need to be re-optimized during the iterative re-optimization procedure. We supplement the formal definitions as follows: given a solution $\mathcal{R}_{t}$ at iterative step $t$, an edge $e\in\mathcal{R}$ is unstable if $e\notin\mathcal{R}_{t+1}$ or $e\notin\mathcal{R}_{t+2}$, …, or $e\notin\mathcal{R}_{t+k}$. When we generate the labels for training, we use a lookahead backbone solver for detecting unstable edges, which equivalently sets $k=1$. An edge is a stable edge if it’s not an unstable edge.

### A.2 Capacitated Vehicle Routing Problem

Given a complete graph $G=(V,E)$ where $V=\{x_{0},x_{1},\ldots,x_{n}\}$ is the set of nodes with node $x_{0}$ representing the depot and nodes $x_{1}$ to $x_{n}$ representing customers. Each customer $i$ has a demand $d_{i}>0$, and each edge $e_{i,j}\in E$ has an associated cost representing the travel distance or travel time between nodes $x_{i}$ and $x_{j}$. A fleet of homogeneous vehicles, each with capacity $C$, is available at the depot. The objective is to find a set of routes that minimizes the total travel cost, subject to: (i) each route starts and ends at the depot, (ii) each customer is visited exactly once, (iii) the total demand of customers on each route does not exceed vehicle capacity $C$.

### A.3 Vehicle Routing Problem with Time Windows

Given a complete graph $G=(V,E)$ where $V=\{x_{0},x_{1},\ldots,x_{n}\}$ is the set of nodes with node $x_{0}$ representing the depot and nodes $x_{1}$ to $x_{n}$ representing customers. Each customer $i$ has a demand $d_{i}>0$, and each edge $e_{i,j}\in E$ has an associated cost representing the travel distance or travel time between nodes $x_{i}$ and $x_{j}$. Each customer $i$ has a time window $[t^{l}_{i},t^{r}_{i}]$ where $t^{l}_{i}$ is the earliest arrival time and $t^{r}_{i}$ is the latest arrival time, and requires a service time $s_{i}$. A fleet of homogeneous vehicles, each with capacity $C$, is available at the depot. The objective is to find a set of routes that minimizes the total travel cost, subject to: (i) each route starts and ends at the depot, (ii) each customer is visited exactly once, (iii) the total demand of customers on each route does not exceed vehicle capacity $C$, (iv) service at each customer begins within their time window $[t^{l}_{i},t^{r}_{i}]$.

## Appendix B Details of First-Segment-Then-Aggregate (FSTA)

### B.1 More discussions on FSTA

#### B.1.1 Visualization of Unstable Edge Patterns

Figure: (a) Random instance 1 at step 1
Refer to caption: https://arxiv.org/html/2507.01037/figures/subopt_dis_ins1_step1.jpg

In this section, we provide visualization and analysis of unstable edge distribution patterns, which serve as foundational motivation for our L2Seg approach. We examine unstable edges on three randomly selected CVRP1k instances solved iteratively using LKH-3. In these visualizations, red dashed lines represent unstable edges, and yellow stars indicate depot locations.

Our visualization reveals two key observations: (1) The number of unstable edges generally decreases as optimization progresses, with more and more edges remaining unchanged between iterations; (2) Edges at route boundaries exhibit higher stability, while unstable edges predominantly concentrate within route interiors. Despite these discernible spatial patterns, no simple heuristic rule appears sufficient to reliably predict unstable edges, as they can be distributed across the start, middle, and end segments of each tour. This complexity motivates our development of L2Seg, a learning-based method designed to capture these intricate patterns more effectively.

#### B.1.2 Visualization of Applying FSTA on
One CVRP Instance

Figure: (a) The original CVRP1k instance
Refer to caption: https://arxiv.org/html/2507.01037/figures/FSTA_eg_ori.jpg

To provide a concrete illustration of our FSTA methodology, we present an example of its application to CVRP in Figure [9](https://arxiv.org/html/2507.01037v2#A2.F9), which demonstrates the complete FSTA decomposition pipeline (detailed algorithmic specifications are provided in Appendix [B.1.4](https://arxiv.org/html/2507.01037v2#A2.SS1.SSS4)). This example utilizes the lookahead oracle model for unstable edge identification (defined in Appendix [B.1.1](https://arxiv.org/html/2507.01037v2#A2.SS1.SSS1)), employs LKH-3 as the backbone optimization solver, and operates on a representative small-capacity CVRP1k instance to showcase the framework’s efficacy. Red dashed lines indicate detected unstable edges, while blue dashed lines represent re-optimized edges. Note that dual hypernode aggregation substantially reduces the problem size compared to the original instance.

#### B.1.3 Assumption Verification

**Table 5: Oracle Performance on CVRP2k: Time to Reach L2Seg-SYN-LNS Solution Quality**
|  | Oracle (LNS) + | Oracle (LNS) + | Oracle (LNS) + | Oracle + | Ref |
| --- | --- | --- | --- | --- | --- |
|  | perfect recall | 95% recall | 90% recall | 70% recall | (L2Seg-SYN-LNS) |
|  | & TNR | & 95% TNR | & 90% TNR | & 70% TNR |  |
| Obj. | 56.02 | 56.01 | 56.02 | 56.04 | 56.08 |
| Time | 39s | 62s | 119s | 324s | 241s |

In Section [3](https://arxiv.org/html/2507.01037v2#S3), we hypothesized that effective problem reduction can substantially accelerate re-optimization. We empirically validate this by implementing a look-ahead oracle for unstable edge detection. The oracle performs a 1-step re-optimization using LKH-3 and identifies unstable edges $E_{\text{unstable}}$ as those differing between the original and re-optimized solutions. FSTA then constructs a reduced problem instance based on these oracle-identified edges, which is subsequently re-optimized using the LKH-3 backbone solver. As this is an oracle-based evaluation, the time required for look-ahead computation is excluded from performance measurements.

Table [5](https://arxiv.org/html/2507.01037v2#A2.T5) reports the time required to achieve performance equivalent to our learned model on small-capacity CVRP2k instances. Beyond the perfect oracle scenario, we evaluate imperfect oracle configurations where recall and true negative rates fall below 100%. The perfect oracle demonstrates substantially superior efficiency. Performance remains competitive under moderate imperfection levels; however, achieving recall and TNR as high as 90% without oracle access is highly non-trivial. In more practical scenarios, where recall and TNR drop to 70%, the oracle-based approach is outperformed by our L2Seg-SYN-LNS, highlighting the effectiveness of our learned model.

These results provide evidence that accurate identification of unstable edges, coupled with appropriate FSTA-based problem reduction, enables significantly more efficient re-optimization.

#### B.1.4 Details of FSTA Decomposition Framework

In this section, we present the details of the FSTA decomposition framework. Given a routing problem $P$ and an initial solution $\mathcal{R}$, one iterative step of FSTA yields a potentially improved solution $\mathcal{R}_{+}$. The framework comprises five sequential steps (also illustrated in Algorithm [1](https://arxiv.org/html/2507.01037v2#algorithm1) and Figure [2](https://arxiv.org/html/2507.01037v2#S3.F2)):

- 1.
Unstable Edges Detection: We implement effective methods (e.g., our learning-based model L2Seg or random heuristics detailed in Section [5.3](https://arxiv.org/html/2507.01037v2#S5.SS3)) to identify unstable edges $E_{\text{unstable}}$ and obtain the stable edge set $E_{\text{stable}}=E\setminus E_{\text{unstable}}$. This identification challenge is addressed by our L2Seg model, with full details provided in Section [4](https://arxiv.org/html/2507.01037v2#S4) and Appendix [C](https://arxiv.org/html/2507.01037v2#A3).
- 2.
Segment Partitioning: After removing unstable edges $E_{\text{unstable}}$, each route decomposes into multiple disjoint segments consisting of consecutive nodes connected by stable edges. Formally, we segment each route into $(x_{0},S^{i}_{1,j_{1}},S^{i}_{j_{1},j_{2}},...,x_{0})=(x_{0},S^{i}_{(1)},S^{i}_{(2)},...,x_{0})\in R^{i}$, where $x_{0}$ is depot and we simplify the notation by using a single index for segments (note that a segment can consist only one single node).
- 3.
Hypernode Aggregation: We aggregate each segment $S^{i}_{j,k}$ and represent it with either one hypernode ($\tilde{S}^{i}_{j,k}=\{\tilde{x}^{i}_{j,k}\}$) or two hypernodes ($\tilde{S}^{i}_{j,k}=\{\tilde{x}^{i}_{j},\tilde{x}^{i}_{k}\}$) with aggregated attributes. This transformation requires that (our feasibility theorem): (a) the reduced problem remains feasible, and (b) a solution in the aggregated problem can be mapped back to a feasible solution in the original problem. These transformations produce a reduced problem $\tilde{P}$ with corresponding solution $\tilde{\mathcal{R}}$.
- 4.
Re-optimization with Backbone Solvers: We invoke a backbone solver to improve solution $\tilde{\mathcal{R}}$, yielding an enhanced solution $\tilde{\mathcal{R}}_{+}$. While theoretically any solver could serve as the backbone solver, practical acceleration requires solvers capable of effectively leveraging existing solutions (e.g., LKH-3 (Helsgaun, 2017)).
- 5.
Solution Recovery: With the improved solution $\tilde{\mathcal{R}}_{+}$ for the reduced problem $\tilde{P}$, we recover a corresponding solution $\mathcal{R}_{+}$ for the original problem $P$ by expanding each hypernode back into its original segment of nodes. This step relies on our monotonicity theorem, which guarantees that an improved solution in $\tilde{P}$ maps to an improved solution in $P$.

Selection of Hypernode Aggregation Strategies. We analyze the trade-offs between single and dual hypernode aggregation strategies: (1) Dual hypernode aggregation enables bidirectional segment traversal, potentially improving re-optimization efficiency by expanding the solution search space. However, this approach requires enforcing inclusion of the connecting edge between hypernodes, adding algorithmic complexity.
(2) Single hypernode aggregation achieves superior problem size reduction but constrains segment traversal to a fixed direction, thereby restricting the re-optimization search space and potentially limiting performance improvements. Additionally, single hypernode aggregation transforms symmetric routing problems into asymmetric variants, which may compromise the efficiency of existing backbone solvers that are typically optimized for symmetric instances.

Selection of Backbone Solvers.
Our framework is generic to be applied to most existing VRP heuristics by design. In practice, acceleration within our framework requires solvers that can effectively utilize initial solutions as warm starts. Furthermore, if the dual hypernode aggregation is used, the backbone solver needs to fix certain edges during local search. Our framework is readily compatible with a variety of solvers without modifying their source codes, including LKH-3 (Helsgaun, 2017), decomposition-based solvers like LNS (Shaw, 1998), and learning-based methods such as L2D (Li et al., 2021). Incorporating additional solvers such as HGS (Vidal, 2022), would involve extending its current code to accept initial solutions as input, which we leave as future work.
Notably, as demonstrated in Section [5](https://arxiv.org/html/2507.01037v2#S5), our L2Seg-augmented approach with relatively weaker backbone solvers outperforms HGS in multiple CVRP and VRPTW benchmark scenarios.

Applicability to Routing Variants.
FSTA is broadly applicable to routing problem variants that support feasible hypernode aggregation and solution recovery, as ensured by the feasibility and monotonicity conditions established in Section [3](https://arxiv.org/html/2507.01037v2#S3). In Appendix [B.2](https://arxiv.org/html/2507.01037v2#A2.SS2), we formally prove that many routing variants meet these conditions, demonstrating the versatility of our L2Seg framework. Detailed implementation guidelines for applying hypernode aggregation across different routing variants are provided in Appendix [B.1.5](https://arxiv.org/html/2507.01037v2#A2.SS1.SSS5).

Figure: Algorithm 1 Iteratively Re-optimize Routing Problems with FSTA

#### B.1.5 Applying FSTA on Various VRPs

In this section, we present the implementation details of FSTA across diverse routing variants, including the Capacitated Vehicle Routing Problem (CVRP), Vehicle Routing Problem with Time Windows (VRPTW), Vehicle Routing Problem with Backhauls (VRPB), and Single-Commodity Vehicle Routing Problem with Pickup and Delivery (1-VRPPD). Without loss of generality, we denote a segment to be aggregated as $S_{j,k}=(x_{j}\rightarrow\ldots\rightarrow x_{k})$, and its corresponding hypernode representation as either $\tilde{S}_{j,k}=\{\tilde{x}\}$ (single hypernode) or $\tilde{S}_{j,k}=\{\tilde{x}_{j},\tilde{x}_{k}\}$ (dual hypernodes). The implementation specifications are summarized in Table [6](https://arxiv.org/html/2507.01037v2#A2.T6).

CVRP. We provide the formal definition of CVRP in Section [3](https://arxiv.org/html/2507.01037v2#S3). Each node in CVRP is characterized by location and demand attributes. For CVRP, we employ dual hypernode aggregation where location attributes are preserved as $\tilde{x}_{j}=x_{j}$ and $\tilde{x}_{k}=x_{k}$, while demand is equally distributed between hypernodes as $\tilde{d}_{j}=\tilde{d}_{k}=\frac{1}{2}\left(d_{j}+\cdots+d_{k}\right)$. We force the solver to include the edge connecting $\tilde{x}_{j}$ and $\tilde{x}_{k}$ in the solution.

VRPTW. We provide the formal definition of VRPTW in Section [3](https://arxiv.org/html/2507.01037v2#S3). In addition to location and demand attributes, VRPTW instances are characterized by time windows $[t^{l},t^{r}]$ and service time $s$ for each node. For VRPTW, we employ adaptive strategies for hypernode aggregation based on temporal feasibility. We first compute the aggregated time windows $\bar{t}^{l}_{j}$, $\bar{t}^{r}_{j}$ and aggregated service time $\bar{s}_{j}$ using the following recursive formulation:

$$ $\displaystyle\bar{t}_{m}^{l}$ $\displaystyle=$ (3) $\displaystyle\bar{t}_{m}^{r}$ $\displaystyle=$ $\displaystyle\bar{s}_{m}$ $\displaystyle=$ $$

where $[t_{m}^{l},t_{m}^{r}]$ denotes the time window for node $x_{m}$, $s_{m}$ represents the service time at node $x_{m}$, and $\text{dist}(x_{m},x_{m+1})$ is the travel time from node $x_{m}$ to node $x_{m+1}$.

If $\bar{t}^{l}_{j}\leq\bar{t}^{r}_{j}$ (feasible time window), we employ single hypernode aggregation with: $\text{dist}(x_{i},\tilde{x})=\text{dist}(x_{i},x_{j})$, $\text{dist}(\tilde{x},x_{i})=\text{dist}(x_{k},x_{i})$, $\tilde{d}=d_{j}+\cdots+d_{k}$, $\tilde{t}^{l}=\bar{t}^{l}_{j}$, $\tilde{t}^{r}=\bar{t}^{r}_{j}$, and $\tilde{s}=\bar{s}_{j}$.

If $\bar{t}^{l}_{j}>\bar{t}^{r}_{j}$ (temporal infeasible time window), we employ dual hypernode aggregation with: $\tilde{x}_{j}=x_{j}$, $\tilde{x}_{k}=x_{k}$, $\tilde{d}_{j}=\tilde{d}_{k}=\frac{1}{2}(d_{j}+\cdots+d_{k})$, time windows $\tilde{t}_{j}^{l}=0$, $\tilde{t}_{j}^{r}=\bar{t}^{r}_{j}$, $\tilde{t}^{l}_{k}=\bar{t}^{l}_{j}$, $\tilde{t}^{r}_{k}=\infty$, and service times $\tilde{s}_{j}=0$, $\tilde{s}_{k}=\bar{s}_{j}$. We additionally set $\text{dist}(\tilde{x}_{j},\tilde{x}_{k})=0$ and enforce inclusion of the edge connecting $\tilde{x}_{j}$ and $\tilde{x}_{k}$ in the solution.

VRPB. Compared to the CVRP, the VRPB (Goetschalckx and Jacobs-Blecha, 1989) involves serving two types of customers: linehaul customers requiring deliveries from the depot and backhaul customers providing goods to be collected and returned to the depot. The primary constraint is that all linehaul customers must be visited before any backhaul customers on the same route, while ensuring vehicle capacity is never exceeded during either the delivery or pickup phases. We use $b_{i}\in\{0,1\}$ to indicate whether node $i$ is a backhaul customer. For VRPB, we require the edge connecting to a linehaul customer and a backhaul customer included in the $E_{\text{unstable}}$. We employ single hypernode aggregation that $\text{dist}(x_{i},\tilde{x})=\text{dist}(x_{i},x_{j})$, $\text{dist}(\tilde{x},x_{i})=\text{dist}(x_{k},x_{i})$, $\tilde{d}=d_{j}+\cdots+d_{k}$, and $\tilde{b}=b_{j}$ (we require customer being the same type within each segment that $b_{j}=...=b_{k}$).

1-VRPPD. Compared to the CVRP, the 1-VRPPD (Martinovic et al., 2008) deals with customers labeled as either cargo sink ($d_{i}<0$) or cargo source ($d_{i}>0$), depending on their pickup or delivery demand. Along the route of each vehicle, the vehicle could not load negative cargo or cargo exceeding the capacity of the vehicle $C$. For any segment $S_{j,k}$, we define $D^{j}=d_{j}$, $D^{j+1}=d_{j}+d_{j+1}$, …, and $D^{k}=d_{j}+d_{j+1}+...+d_{k}$. We further define $D^{\text{min}}=\min\{0,D_{j},D_{j+1},...\}$ and $D^{\text{max}}=\max\{0,D_{j},D_{j+1},...\}$. For 1-VRPPD, we require three hypernodes $\tilde{x}_{j}=x_{j}$, $\tilde{x}_{\text{mid}}$, and $\tilde{x}_{k}=x_{k}$, where the distances from $\tilde{x}_{\text{mid}}$ to $\tilde{x}_{j}$ or $\tilde{x}_{k}$ are 0, and infinity for the other hypernodes. For the aggregated demands, $\tilde{d}_{j}=D^{\text{min}}$, $\tilde{d}_{\text{mid}}=D^{\text{max}}-D^{\text{min}}$, and $\tilde{d}_{k}=D^{k}-D^{\text{max}}-D^{\text{min}}$. Additional constraints are added to ensure the directed edges $\tilde{x}_{j}\rightarrow\tilde{x}_{\text{mid}}\rightarrow\tilde{x}_{k}$ are included in the solutions.

**Table 6: Implementation specifications of FSTA hypernode aggregation for CVRP, VRPTW, VRPB variants. Refer to Equation [3](https://arxiv.org/html/2507.01037v2#A2.E3) for the definitions of $\bar{s}_{j}$, $\bar{t}^{l}_{j}$ and $\bar{t}^{r}_{j}$.**
| CVRP |  |  |  |  |
| --- | --- | --- | --- | --- |
| Type | Condition | Attribute | Aggregation | Additional Constraints / Settings |
| Two Hypernodes | Always | Location/Distance | $\tilde{x}_{j}=x_{j}$ | Include edge $\tilde{x}_{j}\rightarrow\tilde{x}_{k}$ in the solution |
|  | $\tilde{x}_{k}=x_{k}$ |  |  |  |
| Demand | $\tilde{d}_{j}=\tilde{d}_{k}=\frac{1}{2}(d_{j}+\cdots+d_{k})$ |  |  |  |
| VRPTW |  |  |  |  |
| Type | Condition | Attribute | Aggregation | Additional Constraints / Settings |
| One Hypernode | $\bar{t}^{l}_{j}\leq\bar{t}^{r}_{j}$ | Location/Distance | $\text{dist}(x_{i},\tilde{x})=\text{dist}(x_{i},x_{j})$, | None |
|  | $\text{dist}(\tilde{x},x_{i})=\text{dist}(x_{k},x_{i})$ |  |  |  |
| Demand | $\tilde{d}=d_{j}+\cdots+d_{k}$ |  |  |  |
| Service Time | $\tilde{s}=\bar{s}_{j}$ |  |  |  |
| Time Windows | $\tilde{t}^{l}=\bar{t}^{l}_{j}$, $\tilde{t}^{r}=\bar{t}^{r}_{j}$ |  |  |  |
| Two Hypernodes | $\bar{t}^{l}_{j}>\bar{t}^{r}_{j}$ | Location/Distance | $\tilde{x}_{j}=x_{j}$, $\tilde{x}_{k}=x_{k}$ | Include edge $\tilde{x}_{j}\rightarrow\tilde{x}_{k}$ in solution; set $\text{dist}(\tilde{x}_{j},\tilde{x}_{k})=0$ |
| Demand | $\tilde{d}_{j}=\tilde{d}_{k}=\frac{1}{2}(d_{j}+\cdots+d_{k})$ |  |  |  |
| Service Time | $\tilde{s}_{j}=0$, $\tilde{s}_{k}=\bar{s}_{j}$ |  |  |  |
| Time Windows | $\tilde{t}_{j}^{l}=0$, $\tilde{t}_{j}^{r}=\bar{t}^{r}_{j}$, $\tilde{t}^{l}_{k}=\bar{t}^{l}_{j}$, $\tilde{t}^{r}_{k}=\infty$ |  |  |  |
| VRPB |  |  |  |  |
| Type | Condition | Attribute | Aggregation | Additional Constraints / Settings |
| One Hypernode | Always | Location/Distance | $\text{dist}(x_{i},\tilde{x})=\text{dist}(x_{i},x_{j})$, | Require $b_{j}=\cdots=b_{k}$ (same customer type) during Unstable Edges Detection Stage |
|  | $\text{dist}(\tilde{x},x_{i})=\text{dist}(x_{k},x_{i})$ |  |  |  |
| Demand | $\tilde{d}=d_{j}+\cdots+d_{k}$ |  |  |  |
| Is backhaul | $\tilde{b}=b_{j}$ |  |  |  |
| 1-VRPPD |  |  |  |  |
| Type | Condition | Attribute | Aggregation | Additional Constraints / Settings |
| Three Hypernodes | Always | Location/Distance | $\tilde{x}_{j}=x_{j}$, $\tilde{x}_{k}=x_{k}$ | Include edges $\tilde{x}_{j}\rightarrow\tilde{x}_{\text{mid}}\rightarrow\tilde{x}_{k}$ in the solution |
|  | $\text{dist}(\tilde{x}_{j},\tilde{x}_{\text{mid}})=\text{dist}(\tilde{x}_{\text{mid}},\tilde{x}_{k})=0$ |  |  |  |
|  | $\tilde{x}_{\text{mid}}$ only connects to $\tilde{x}_{j}$ and $\tilde{x}_{k}$ |  |  |  |
| Demand | $\tilde{d}_{j}=D^{\text{min}}$, $\tilde{d}_{\text{mid}}=D^{\text{max}}-D^{\text{min}}$, |  |  |  |
|  | $\tilde{d}_{k}=D^{k}-D^{\text{max}}-D^{\text{min}}$ |  |  |  |

### B.2 Proof of FSTA

Theorem. (Feasibility) If the aggregated solution $\tilde{\mathcal{R}}_{+}$ is a feasible solution to the aggregated problem, then $\mathcal{R}_{+}$ is a feasible solution to the original, non-aggregated problem.
(Monotonicity) Let $\tilde{\mathcal{R}}^{1}_{+}$ and $\tilde{\mathcal{R}}^{2}_{+}$ be two feasible solutions to the aggregated problem, with $f(\tilde{\mathcal{R}}^{1}_{+})\leq f(\tilde{\mathcal{R}}^{2}_{+})$, where $f(\cdot)$ denotes the objective function (total travel cost). Then, for the associated solution in the original space, we also have $f(\mathcal{R}^{1}_{+})\leq f(\mathcal{R}^{2}_{+})$.

Proof Structure and Notation. Without loss of generality, we consider a single-route solution containing one segment $S_{j,k}=\left(x_{j}\rightarrow\cdots\rightarrow x_{k}\right)$ with more than one node, i.e., the solution $\mathcal{R}$ contains route $R=(x_{0}\rightarrow x_{1}\rightarrow\cdots\rightarrow S_{j,k}\rightarrow x_{k+1}\rightarrow\cdots\rightarrow x_{0})$. We define the aggregated problem with node set $\tilde{V}=\{x_{0}\}\cup\{x_{p}\}_{\begin{subarray}{c}p<j\text{ or}\\
p>k\end{subarray}}\cup\{\tilde{S}_{j,k}\}$, where nodes outside the segment retain their original representation, ensuring their feasibility by construction. Since we enforce the inclusion of the edge connecting $\tilde{x}_{j}$ and $\tilde{x}_{k}$ in dual hypernode aggregation within solution $\tilde{\mathcal{R}}_{+}$, the segment $\tilde{S}_{j,k}$ must be incorporated into some route $\tilde{R}^{*}_{+}\in\tilde{\mathcal{R}}_{+}$ for both hypernode aggregation strategies. We denote the improved route containing this segment after mapping back to the original problem as $R^{*}_{+}$.

We present the segment aggregation strategies for different routing variants below, followed by proofs of feasibility and monotonicity for the aggregation scheme. Note that the following analysis naturally extends to multi-route solutions with multiple segments per route.

#### B.2.1 CVRP

Aggregation Strategy (Two Hypernodes). The detailed implementation of FSTA on CVRP can be found in Appendix [B.1.5](https://arxiv.org/html/2507.01037v2#A2.SS1.SSS5) and Table [6](https://arxiv.org/html/2507.01037v2#A2.T6). Notice that one single hypernode aggregation is also applicatable for CVRP, and $\tilde{d}_{j}$,$\tilde{d_{k}}$ could take other values as long as $\tilde{d}_{j}+\tilde{d_{k}}=d_{j}+...+d_{k}$.

Feasibility Proof [Capacity Constraint]. Notice that since $\tilde{d}_{j}+\tilde{d_{k}}=d_{j}+...+d_{k}$, we have:

$$ $\begin{split}\sum_{x_{i}\in\tilde{R}^{*}_{+}}d_{i}&=\sum_{x_{i}\in\tilde{R}^{*}_{+}\setminus\tilde{S}_{j,k}}d_{i}+\,\,\,\,\,\tilde{d}_{j}+\tilde{d}_{k}\\ &=\sum_{x_{i}\in R^{*}_{+}\setminus S_{j,k}}d_{i}+\,\,\,\,\,d_{j}+...+d_{k}=\sum_{x_{i}\in R^{*}_{+}}d_{i}\end{split}$ (4) $$

Thus, we have:

$$ $\begin{split}\sum_{x_{i}\in\tilde{R}^{*}_{+}}d_{i}\leq C\Rightarrow\sum_{x_{i}\in R^{*}_{+}}d_{i}\leq C\end{split}$ (5) $$

Then, we have a feasible $\tilde{\mathcal{R}}_{+}\Rightarrow$ a feasible $\mathcal{R}_{+}$.

∎

Monotonicity Proof. Notice that

$$ $\begin{split}f(\tilde{\mathcal{R}}_{+})&=f(\tilde{\mathcal{R}}_{+}\setminus\{\tilde{R}^{*}_{+}\})+f(\{\tilde{R}^{*}_{+}\})=f(\mathcal{R}_{+}\setminus\{R^{*}_{+}\})+f(\{\tilde{R}^{*}_{+}\})\\ &=f(\mathcal{R}_{+}\setminus\{R^{*}_{+}\})+f(\{R^{*}_{+}\})-\sum_{j\leq q<k}dist(x_{q},x_{q+1})\,\,\,\,\,+dist(\tilde{x}_{j},\tilde{x}_{k})\\ &=f(\mathcal{R}_{+})+\text{Const}|_{S_{j,k}}\end{split}$ (6) $$

where $\text{Const}|_{S_{j,k}}$ is a constant once the segment $S_{j,k}$ is decided. Therefore, we have:

$$ $\begin{split}f(\tilde{\mathcal{R}}^{1}_{+})\leq f(\tilde{\mathcal{R}}^{2}_{+})\Rightarrow f(\mathcal{R}^{1}_{+})+\text{Const}|_{S_{j,k}}\leq f(\mathcal{R}^{2}_{+})+\text{Const}|_{S_{j,k}}\Rightarrow f(\mathcal{R}^{1}_{+})\leq f(\mathcal{R}^{2}_{+})\end{split}$ (7) $$

∎

We note that the feasibility proof for capacity constraint and the monotonicity proof could be easily extended to the single hypernodes aggregation.

#### B.2.2 VRPTW

Aggregation Strategy (Mixed Strategies). The detailed implementation of FSTA on VRPTW can be found in Appendix [B.1.5](https://arxiv.org/html/2507.01037v2#A2.SS1.SSS5) and Table [6](https://arxiv.org/html/2507.01037v2#A2.T6). We denote $s^{*}_{m}=s_{m}+\text{dist}(x_{m},x_{m+1})$ for $j\leq m<k$ and $s^{*}_{k}=s_{k}$. We further set the service time by $\tilde{s}_{m}=\sum\limits_{m\leq q\leq k}s^{*}_{q}$, and we repeat the temporal time window $[\bar{t}^{l}_{j},\bar{t}^{r}_{j}]$ (which could be infeasible) defined by the following recursive relationship:

$$ $\displaystyle\bar{t}_{m}^{l}$ $\displaystyle=$ (8) $\displaystyle\bar{t}_{m}^{r}$ $\displaystyle=$ $$

where $[t_{m}^{l},t_{m}^{r}]$ is the time window for a node $x_{m}$, $s_{m}$ is the service time at node $x_{m}$ and $\text{dist}(x_{m},x_{m+1})$ is the time to travel from node $x_{m}$ to node $x_{m+1}$.

Feasibility Proof [Time Window Constraint]. We first prove for the condition that the temporal time window $[\bar{t}^{l}_{j},\bar{t}^{r}_{j}]$ is feasible ($\bar{t}^{l}_{j}<\bar{t}^{r}_{j}$) and single hypernode aggregation is applied. Then, we extend to the infeasible temporal time window condition where dual hypernode aggregation is applied.

Condition of Feasible Temporal Time Windows (One Hypernode). We present an inductive proof based on the segment length. Given a feasible solution $\tilde{\mathcal{R}}_{+}$ for the aggregated problem, we show the following two conditions of the corresponding non-aggregated solution $\mathcal{R}_{+}$ to satisfy the time window constraint:

- •
Condition (1): We visit each node $x_{m}$ before the end of its time window $t_{m}^{r}$.
- •
Condition (2): The total time we spent visiting the entire segment is the same in both aggregated and non-aggregated representations.

Proof of Condition (1):

- •
Base case (segment length = 1). Suppose the segment $S_{k,k}=\left(x_{k}\right)$ contains a single node $x_{k}$. Then the aggregated problem is identical to the non-aggregated problem by construction, so condition (1) is trivially satisfied.
- •
Inductive Step (segment length = $(k-m)+1>1$). textit the aggregation of the segment $S_{m+1,k}=\left(x_{m+1}\rightarrow...\rightarrow x_{k}\right)$ into $\tilde{S}_{m+1,k}=\{\tilde{x}_{m+1,k}\}$ satisfies condition (1). We want to show that the aggregation of the segment $S_{m,k}=\left(x_{m}\rightarrow...\rightarrow x_{k}\right)$ into $\tilde{S}_{m,k}=\{\tilde{x}_{m,k}\}$ also satisfies condition (1).
Since $\tilde{\mathcal{R}}_{+}$ is a feasible solution for the aggregated problem, we will visit the hypernode $\tilde{x}_{m,k}$ before the end of its time window $\bar{t}^{r}_{m}=\min\{t_{m}^{r},\bar{t}_{m+1}^{r}-s^{*}_{m}\}$. Corresponding, in the associated non-aggregated solution, we visit the node $x_{m}$ before its time limit $t_{m}^{r}$, hence satisfying condition (1) for the node $x_{m}$. Furthermore, in the associated non-aggregated solution, we visit the next node $x_{m+1}$ before time $\bar{t}^{r}_{m}+s^{*}_{m}\leq\bar{t}_{m+1}^{r}$. Based on the inductive hypothesis, condition (1) holds for the rest of the segment $\left(x_{m+1}\rightarrow...\rightarrow x_{k}\right)$ if we arrive at node $x_{m+1}$ before its end time. Hence, condition (1) holds for the whole segment $S_{m,k}=(x_{m}\rightarrow x_{m+1}\rightarrow...\rightarrow x_{k})$.

Figure: Figure 10: This illustration demonstrates the temporal dynamics of the aggregated segment. The left panel shows the time function characterized by a piecewise linear structure: initially decreasing with slope -1, then transitioning to a constant value corresponding to the aggregated left time window boundary. The right panel presents two distinct scenarios that characterize the relationship between the aggregated left time window ($\bar{t}^{l}_{m}$) and the individual non-aggregated left time windows ($t^{l}_{m}$).
Refer to caption: https://arxiv.org/html/2507.01037/x5.png

Proof of Condition (2): For all $m$, suppose we arrive at the hypernode $\tilde{x}_{m,k}$ at time $t\leq\bar{t}_{m}^{r}$ in the aggregated solution. By definition, the total time spent on the aggregated segment (sum of the waiting time, service time, and the travel time) can be written as the following linear function with $-1$ slope as shown in the first figure in Figure [10](https://arxiv.org/html/2507.01037v2#A2.F10).

$$ $\tilde{g}_{m}(t)=\begin{cases}\tilde{s}_{m}&t\geq\bar{t}_{m}^{l}\\ \bar{t}_{m}^{l}-t+\tilde{s}_{m}&t<\bar{t}_{m}^{l}.\end{cases}$ (9) $$

Note: the first condition $t\geq\bar{t}_{m}$ means we do not need to wait at any node in the segment $S_{m,k}$, and the second condition means we need to wait at some node in the segment $S_{m,k}$.

It suffices to show that the total time spent on the non-aggregated segment also follows the same function. Again, we prove this by induction.

- •
Base case (segment length = 1). Suppose the segment $S_{k,k}=\left(x_{k}\right)$ contains a single node $x_{k}$. Then the aggregated problem is identical to the non-aggregated problem by construction, so the total time spent on the non-agggregated segment is exactly Eq. [9](https://arxiv.org/html/2507.01037v2#A2.E9) with $m=k$.
- •
Inductive Step (segment length = $(k-m)+1>1$). Again, suppose the total time spent on the segment $S_{m+1,k}=\left(x_{m+1}\rightarrow...\rightarrow x_{k}\right)$ into $\tilde{S}_{m+1,k}=\{\tilde{x}_{m+1,k}\}$ satisfies the function
$g_{m+1}(t)=\tilde{g}_{m+1}(t)=\begin{cases}\tilde{s}_{m+1}&t\geq\bar{t}_{m+1}^{l}\\
\bar{t}_{m+1}^{l}-t+\tilde{s}_{m+1}&t<\bar{t}_{m+1}^{l}\end{cases}$
(10)
We now show the total time function $g_{m}(t)$ for the segment $S_{m,k}=\left(x_{m}\rightarrow...\rightarrow x_{k}\right)$ also equals $\tilde{g}_{m}(t)$.
By definition of the non-aggregated segment, depending on whether we need to wait at the first node $x_{m}$, we have:
$g_{m}(t)=\begin{cases}s^{*}_{m}+g_{m+1}(t+s^{*}_{m})&t\geq t_{m}^{l}\\
t_{m}^{l}-t+s^{*}_{m}+g_{m+1}(t_{m}^{l}+s^{*}_{m})&t<t_{m}^{l}.\end{cases}$
(11)
Note: the first condition $t\geq t_{m}^{l}$ means we do not need to wait at the first node $x_{m}$, and the second condition $t<t_{m}^{l}$ means we need to wait at the first node $x_{m}$.
We split the discussion into the following two cases, based on whether we need to wait at any node along the segment $S_{m+1,k}$, if we leave node $x_{m}$ at $t_{m}^{l}$:
1.
$t_{m}^{l}+s^{*}_{m}\geq\bar{t}_{m+1}^{l}$. In this case, $t_{m}^{l}\geq\bar{t}_{m+1}^{l}-s^{*}_{m}$, and hence $\bar{t}_{m}^{l}=\max\{t_{m}^{l},\bar{t}_{m+1}^{l}-s^{*}_{m}\})=t_{m}^{l}$ as shown in case 1 of Figure [10](https://arxiv.org/html/2507.01037v2#A2.F10). Hence, we have
$g_{m}(t)=\begin{cases}s^{*}_{m}+g_{m+1}(t+s^{*}_{m})&t\geq\bar{t}_{m}^{l}\\
t_{m}^{l}-t+s^{*}_{m}+g_{m+1}(t_{m}^{l}+s^{*}_{m})&t<\bar{t}_{m}^{l}.\end{cases}$
(12)
By inductive hypothesis, we have
$g_{m+1}(t+s^{*}_{m})=\tilde{s}_{m+1},\quad t\geq t_{m}^{l}=\bar{t}_{m}^{l},$
as in this case $t+s^{*}_{m}\geq t_{m}^{l}+s^{*}_{m}\geq\bar{t}_{m+1}^{l}$.
Hence, we have
$\displaystyle g_{m}(t)$
$\displaystyle=$
(13)
$\displaystyle==\tilde{g}_{m}(t).$
where we apply the definition of $\tilde{s}_{m}=s^{*}_{m}+\tilde{s}_{m+1}$.
2.
$t_{m}^{l}+s^{*}_{m}<\bar{t}_{m+1}^{l}$. In this case, $\bar{t}^{l}_{m+1}-s^{*}_{m}>t_{m}^{l}$, and hence $\bar{t}_{m}^{l}=\max\{t_{m}^{l},\bar{t}^{l}_{m+1}-s^{*}_{m}\}=\bar{t}^{l}_{m+1}-s^{*}_{m}$ as shown in case 2 of Figure [10](https://arxiv.org/html/2507.01037v2#A2.F10).
By inductive hypothesis, we have
$\displaystyle g_{m+1}(t_{m}^{l}+s^{*}_{m})$
$\displaystyle=\bar{t}_{m+1}^{l}-(t_{m}^{l}+s^{*}_{m})+\tilde{s}_{m+1}$
(14)
$\displaystyle=\bar{t}_{m}^{l}-t_{m}^{l}+\tilde{s}_{m+1}$
We also have, for all $t\geq t_{m}^{l}$,
$\displaystyle\quad g_{m+1}(t+s^{*}_{m})$
(15)
$\displaystyle=$
$\displaystyle=$
As a result, we have
$\displaystyle g_{m}(t)$
$\displaystyle=$
(16)
$\displaystyle=$
$\displaystyle==\tilde{g}_{m}(t).$

Condition of Infeasible Temporal Time Windows (Two Hypernodes). In our time window aggregation, $\bar{t}^{l}_{j}$ is responsible for the time expenditure and $\bar{t}^{r}_{j}$ is responsible for feasibility. In this case, we have $\bar{t}^{l}_{j}>\bar{t}^{r}_{j}$, which indicates that to maintain feasibility along the segment, one must arrive at the segment before the aggregated start time $\bar{t}^{l}_{j}$, and since one arrives earlier, one must wait at some node within the segment. Since $\bar{t}^{l}_{j}>\bar{t}^{r}_{j}$ is not permitted according to the definition of VRPTW, we then utilize one additional hypernode to increase the representational capacity such that the first hypernode handles the feasibility component ($\bar{t}^{r}_{j}$), and the second hypernode handles the travel time component ($\bar{t}^{l}_{j}$). Specifically, $\tilde{t}_{j}^{l}=0$, $\tilde{t}_{j}^{r}=\bar{t}^{r}_{j}$, $\tilde{t}^{l}_{k}=\bar{t}^{l}_{j}$, $\tilde{t}^{r}_{k}=\infty$ and $\tilde{s}_{j}=0$, $\tilde{s}_{k}=\bar{s}_{j}$ with the additional constraint that $\text{dist}(\tilde{x}_{j},\tilde{x}_{k})=0$.

For time window feasibility (Condition (1)), since $\tilde{t}_{j}^{r}=\bar{t}^{r}_{j}$, the vehicle must serve the segment before $\bar{t}^{r}_{j}$, ensuring the feasibility of serving each customer in the non-aggregated problem. For travel time equivalence (Condition (2)), the time expended before reaching the second node is $\tilde{s}_{j}+\text{dist}(\tilde{x}_{j},\tilde{x}_{k})=0$. Namely, after the vehicle arrives at the segment at time $t$, the travel time is entirely determined by $\tilde{t}^{l}_{k}=\bar{t}^{l}_{j}$ and $\tilde{s}_{k}=\bar{s}_{j}$, whereby in the feasible temporal time window situation, the travel time equivalence is demonstrated.

We complete the time window constraint feasibility proof for VRPTW for both aggregation strategies across all conditions.

∎

Monotonicity Proof. For the dual hypernode aggregation, please refer to the Monotonicity Proof in [B.2.1](https://arxiv.org/html/2507.01037v2#A2.SS2.SSS1). For the single hypernode aggregation, notice that

$$ $\begin{split}f(\tilde{\mathcal{R}}_{+})&=f(\tilde{\mathcal{R}}_{+}\setminus\{\tilde{R}^{*}_{+}\})+f(\{\tilde{R}^{*}_{+}\})=f(\mathcal{R}_{+}\setminus\{R^{*}_{+}\})+f(\{\tilde{R}^{*}_{+}\})\\ &=f(\mathcal{R}_{+}\setminus\{R^{*}_{+}\})+f(\{R^{*}_{+}\})-\sum_{j\leq q<k}dist(x_{q},x_{q+1})\\ &=f(\mathcal{R}_{+})+\text{Const}|_{S_{j,k}}\end{split}$ (17) $$

where $\text{Const}|_{S_{j,k}}$ is a constant once the segment $S_{j,k}$ is decided. Therefore, we have:

$$ $\begin{split}f(\tilde{\mathcal{R}}^{1}_{+})\leq f(\tilde{\mathcal{R}}^{2}_{+})\Rightarrow f(\mathcal{R}^{1}_{+})+\text{Const}|_{S_{j,k}}\leq f(\mathcal{R}^{2}_{+})+\text{Const}|_{S_{j,k}}\Rightarrow f(\mathcal{R}^{1}_{+})\leq f(\mathcal{R}^{2}_{+})\end{split}$ (18) $$

∎

#### B.2.3 VRPB

Aggregation Strategy (One Hypernode). The detailed implementation of FSTA on VRPB can be found in Appendix [B.1.5](https://arxiv.org/html/2507.01037v2#A2.SS1.SSS5) and Table [6](https://arxiv.org/html/2507.01037v2#A2.T6).

Feasibility Proof [Backhaul Constraint]. Without loss of generality, we assume all nodes within the segment $S_{j,k}$ are backhaul customers ($b_{j}=...=b_{k}=1$). Notice that since $\tilde{d}=d_{j}+...+d_{k}$, for the backhaul stage, we have:

$$ $\begin{split}\sum_{x_{i}\in\tilde{R}^{*}_{+}\,\,\text{and}\,\,b_{i}=1}d_{i}&=\sum_{x_{i}\in\tilde{R}^{*}_{+}\setminus\tilde{S}_{j,k}\,\,\text{and}\,\,b_{i}=1}d_{i}+\tilde{d}\\ &=\sum_{x_{i}\in R^{*}_{+}\setminus S_{j,k}\,\,\text{and}\,\,b_{i}=1}d_{i}+\,\,\,\,\,d_{j}+...+d_{k}=\sum_{x_{i}\in R^{*}_{+}\,\,\text{and}\,\,b_{i}=1}d_{i}\end{split}$ (19) $$

For the linehaul stage, we have:

$$ $\begin{split}\sum_{x_{i}\in\tilde{R}^{*}_{+}\,\,\text{and}\,\,b_{i}=0}d_{i}\,\,\,\,=\sum_{x_{i}\in R^{*}_{+}\,\,\text{and}\,\,b_{i}=0}d_{i}\end{split}$ (20) $$

Thus, we have:

$$ $\begin{split}\sum_{x_{i}\in\tilde{R}^{*}_{+}\,\,\text{and}\,\,b_{i}=0}d_{i}\leq C\,\,\,\,\,&\Rightarrow\sum_{x_{i}\in R^{*}_{+}\,\,\text{and}\,\,b_{i}=0}d_{i}\leq C\\ \sum_{x_{i}\in\tilde{R}^{*}_{+}\,\,\text{and}\,\,b_{i}=1}d_{i}\leq C\,\,\,\,\,&\Rightarrow\sum_{x_{i}\in R^{*}_{+}\,\,\text{and}\,\,b_{i}=1}d_{i}\leq C\end{split}$ (21) $$

Then, we have a feasible $\tilde{\mathcal{R}}_{+}\Rightarrow$ a feasible $\mathcal{R}_{+}$.

∎

Monotonicity Proof. Please refer to the monotonicity proof of VRPTW in Appendix [B.2.2](https://arxiv.org/html/2507.01037v2#A2.SS2.SSS2).

#### B.2.4 1-VRPPD.

Aggregation Strategy (Three Hypernodes). The detailed implementation of FSTA on 1-VRPPD can be found in Appendix [B.1.5](https://arxiv.org/html/2507.01037v2#A2.SS1.SSS5) and Table [6](https://arxiv.org/html/2507.01037v2#A2.T6).

Feasibility Proof [1-Commodity Pickup and Delivery Constraint].
A feasible $\tilde{\mathcal{R}}_{+}$ indicates that whenever the vehicle is traveling an aggregated segment $\tilde{S}_{j,k}$, denoted the starting load of the vehicle to be $d_{\text{st}}$ and ending load of the vehicle to be $d_{\text{ed}}$, we have:

$$ $\begin{split}0\leq d_{\text{st}}+D^{\text{min}}\leq C\\ 0\leq d_{\text{st}}+D^{\text{min}}+D^{\text{max}}-D^{\text{min}}\leq C\\ \end{split}$ (22) $$

which requires $-D^{\text{min}}\leq d_{\text{st}}\leq C-D^{\text{max}}$ and $d_{\text{ed}}=d_{\text{st}}+D^{k}$.

On the other hand, a feasible solution $\mathcal{R}_{+}$ indicates that whenever the vehicle is traveling a segment $S_{j,k}$, denoted the starting load of the vehicle to be $d_{\text{st}}$ and ending load of the vehicle to be $d_{\text{ed}}$, we have:

$$ $0\leq d_{\text{st}}+D^{i}\leq C,\,\,\,\forall i$ (23) $$

which also requires $-D^{\text{min}}\leq d_{\text{st}}\leq C-D^{\text{max}}$ and $d_{\text{ed}}=d_{\text{st}}+D^{k}$.
Then, we have a feasible $\tilde{\mathcal{R}}_{+}\Rightarrow$ a feasible $\mathcal{R}_{+}$.

∎

Monotonicity Proof. As $\text{dist}(\tilde{x}_{j},\tilde{x}_{\text{mid}})=\text{dist}(\tilde{x}_{\text{mid}},\tilde{x}_{k})=0$, we can eliminate the middle hypernode and use a two-hypernode representation when calculating the routing objective. Please refer to the monotonicity proof of CVRP in Appendix [B.2.1](https://arxiv.org/html/2507.01037v2#A2.SS2.SSS1) for the monotonicity proof of two-hypernode representation.

## Appendix C L2Seg Details

### C.1 Comparative Analysis of L2Seg Against Existing Methods

Comparisons with Large Neighborhood Search (LNS). (1)
LNS (Large Neighborhood Search) operates within a bounded local neighborhood. The algorithm selects a specific region, destroys elements within that boundary, and rebuilds only that portion while keeping the rest of the solution intact. For instance, in Li et al. (2021), LNS selects 3-5 subroutes as its neighborhood, modifying only these routes while leaving all others completely unchanged. There is a clear demarcation between the modified neighborhood and the preserved structure. (2)
FSTA (our method), in contrast, operates more globally across the entire solution. It can break existing edges and aggregate segments throughout all subroutes simultaneously, without any predefined neighborhood boundaries. The modifications are distributed across the entire solution rather than confined to a local region, which represents a fundamental departure from existing LNS to more efficiently guide the search. We note that such a flexible framework would not be possible without the proposed ML component, which also constitutes the core novelty and contribution of our work to the field. (3) Moreover, FSTA and LNS are complementary: FSTA can be applied on top of LNS, where LNS first selects a large neighborhood, then FSTA fixes stable edges globally within that selected region.

Comparisons with Evolutionary Algorithms. L2Seg framework and evolutionary algorithms (Vidal, 2022)) approach the preservation of solution components from different angles and with distinct goals, and are not interchangeable in use. Evolutionary algorithms (Vidal, 2022)) rely on crossover to merge relatively “good” components from different parents, aiming to promote diversity and generate promising offspring, while our L2Seg framework introduces a learning-guided mechanism to detect unstable edges and aggregates stable edge sequences into hypernodes, enabling a new form of segment-based decomposition that improves scalability and efficiency.

Comparisons with Path Decomposition Method. (1) Firstly, path decomposition relies on geometric heuristics (e.g., clustering routes by barycenter distances) to identify decomposition boundaries. In contrast, L2Seg employs deep learning models (synergistic NAR-AR architecture) to intelligently predict which segments should be aggregated, capturing complex patterns that simple heuristics cannot identify. We also propose a novel learning-guided framework with bespoke training and inference processes that are unique to the machine learning method. (2) Secondly, while some prior work explores similar decomposition ideas (e.g., on CVRP only), we are the first to study FSTA decomposition theoretically, providing formal definitions, feasibility theorems, and monotonicity guarantees for various VRPs. (3) Lastly, we empirically demonstrate that by leveraging deep learning in our L2Seg framework, our method consistently achieves significant speedups on state-of-the-art backbones. This provides new insights for the community, highlighting the power of learning-guided optimization in accelerating combinatorial solvers.

Comparisons with Previous Learning-based Framework L2D (Li et al., 2021). (1) Different from the sub-route level, our method detects unstable edges both within and across sub-routes, enabling more global and flexible decomposition. (2) It optimizes beyond localized neighborhoods by identifying improvements that span multiple distant regions simultaneously. (3) It reduces the size of sub-routes by aggregating stable segments into hypernodes, whereas L2D reduces only the number of sub-routes per iteration. This segment-level aggregation allows more adaptive and coarse-grained reduction, offering higher efficiency and solution quality, while remaining complementary to L2D.

### C.2 Input Feature Design Details

Previous works Kool et al. (2018), Li et al. (2021), Kwon et al. (2020) typically utilize only basic input features for routing problems (xy-coordinates and normalized demands for node features, and edge cost for edge features). While neural networks can potentially learn complex patterns from these basic features, tailored feature engineering may lead to enhanced model performance. As illustrated in Appendix [B.1](https://arxiv.org/html/2507.01037v2#A2.SS1), we observe that detecting unstable edges may depend on better capturing local dependencies. We therefore design enhanced node and edge features for our learning task, as shown in Table [7](https://arxiv.org/html/2507.01037v2#A3.T7). We also include time windows and service time as node features for VRPTWs.

**Table 7: Description of enhanced input features for nodes and edges.**
| Type | Description | Dimension |
| --- | --- | --- |
| Nodes | The xy coordinates | 2 |
| The normalized demand | 1 |  |
| The centroid of the subtour for each node | 2 |  |
| The coordinates of the two nodes connecting to each node | 4 |  |
| The travel cost of the two edges connecting to each node | 2 |  |
| The relative xy coordinates | 2 |  |
| The angles w.r.t. the depot | 1 |  |
| The weighted angles w.r.t. the depot by the distances | 1 |  |
| The distances of the closest 3 neighbor for each node | 3 |  |
| The percentage of the K nearest nodes | 3 |  |
| that are within the same subtour. K=5, 15, 40 |  |  |
|  | The percentage of the K% nearest nodes | 3 |
|  | that are within the same subtour. K=5, 15, 40 |  |
| Edges | The travel cost | 1 |
| Whether each edge is within the current solution | 1 |  |
| The travel cost rank of each edge w.r.t. the corresponding end points | 1 |  |

### C.3 Masking Details

In general, any set of unstable edges could lead to a feasible FSTA problem reduction. However, employing logic-based local search algorithms to select unstable edges can produce more reasonable action space reduction and improved performance. Thus, we design the deletion and insertion stages of L2Seg to emulate a general local search operation.

For the deletion stage, given the current node $x$, we mask out nodes that are: (1) not connected to $x$; or (2) part of an edge that has already been deleted during the current deletion stage. Note that the model may select the special ending node $x_{\text{end}}$ to terminate the decoding sequence.

For the insertion stage, given the current node $x$, we mask out nodes that are: (1) already connected to $x$; (2) endpoints of two newly inserted edges; or (3) the special ending node $x_{\text{end}}$.

### C.4 Training Data Collection Details

In this section, we present pseudocode that demonstrate the process of generating training labels for both NAR and AR models in Algorithm [2](https://arxiv.org/html/2507.01037v2#algorithm2). As a complement to the methodology described in Section [4](https://arxiv.org/html/2507.01037v2#S4), we derive our training data from $N_{\mathcal{P}}$ distinct problem instances and extract labels from the first $T_{IS}$ iterative improvement steps. For the AR labels, which emulate feasible local search operations, each label (representing a sequence of nodes) is associated with a quantifiable improvement in solution quality. We retain only those labels that yield improvements exceeding the threshold $\eta_{\text{improv}}$, and we employ stochastic sampling by accepting labels with probability $\alpha_{AC}$. This selective approach ensures both high-quality training signals and sufficient diversity across problem instances and optimization trajectories within the same training budget.

Figure: Algorithm 2 Training Data Generation

### C.5 Inference Details

In this section, we present the pseudocode that delineates the inference processes of L2Seg-SYN (Algorithm [3](https://arxiv.org/html/2507.01037v2#algorithm3)), L2Seg-NAR (Algorithm [4](https://arxiv.org/html/2507.01037v2#algorithm4)), and L2Seg-AR (Algorithm [5](https://arxiv.org/html/2507.01037v2#algorithm5)). It is important to note that our implementation leverages batch operations for efficient inference across multiple subproblems simultaneously. The K-means clustering algorithm was strategically selected for initial node identification due to its parallelization capabilities. By merging graphs from different subproblems into a unified structure, we can execute the clustering algorithm once for the entire problem space. This parallel clustering approach through K-means significantly enhances decoding efficiency. Notably, within each iterative step, our design requires only a single call of the NAR and AR models, thereby optimizing computational resources.

Figure: Algorithm 3 L2Seg-SYN: Synergized Prediction

Figure: Algorithm 4 L2Seg-NAR: Non-Autoregressive Prediction

Figure: Algorithm 5 L2Seg-AR: Autoregressive Prediction

## Appendix D Experimental and Implementation Details

### D.1 Backbone solvers

LKH-3. The Lin-Kernighan-Helsgaun algorithm (LKH-3) Helsgaun (2017) represents a strong classical heuristic solver for routing problems, which is widely used in NCO for benchmark. It employs sophisticated $k$-opt moves and effective neighborhood search strategies. For our experiments, we impose time limits rather than local search update limits: 150s and 240s for large-capacity CVRP2k and CVRP5k, respectively, and 2m, 4m, and 10m for VRPTW1k, VRPTW2k, and VRPTW5k, respectively. For small-capacity CVRPs, we adopt the results reported in Zheng et al. (2024).

LNS. Local Neighborhood Search (LNS) Shaw (1998) is a powerful decomposition-based metaheuristic that iteratively improves solutions by destructively and constructively exploring defined search neighborhoods. We implement LNS following the approach in Li et al. (2021), where neighborhoods consisting of three adjacent subroutes are randomly selected for re-optimization. We establish time limits of 150s and 240s for large-capacity CVRP2k and CVRP5k, respectively; 2.5m, 4m, and 5m for small-capacity CVRP1k, CVRP2k, and CVRP5k, respectively; and 2m, 4m, and 10m for VRPTW1k, VRPTW2k, and VRPTW5k, respectively. LKH-3 serves as the backbone solver with a 1,000 per-step local search updates limit.

L2D. Learning to Delegate (L2D) Li et al. (2021) is the state-of-the-art learning-based optimization framework that integrates neural networks with classical optimization solvers to intelligently delegate subproblems to appropriate solvers. The framework employs a neural network trained to identify the most promising neighborhoods for improvement. For comparative fairness, we apply identical time limits and backbone solver configurations as used in our LNS implementation. When augmented by L2Seg, training proceeds in two stages: we first train the L2D models following the methodology in Li et al. (2021), then train the L2Seg model using the resulting pre-trained L2D models.

Initial Solution Heuristics. For both training data generation and inference, we employ the initial solution heuristic inspired by (Li et al., 2021). Our method partitions nodes according to their angular coordinates with respect to the depot. We begin by selecting a reference node, marking its angle as 0, and incrementally incorporate additional nodes into the same group until the collective demand approaches the capacity threshold ($\alpha_{\text{init}}K_{\text{veh}}C\approx\sum d_{i}$), where approximately $K_{\text{veh}}$ vehicles would be required to service the group. This process continues sequentially, forming new groups until all customers are assigned. Finally, we apply LKH-3 in parallel to solve each subproblem independently. In our implementation, we set $K_{\text{veh}}=6$ and $\alpha_{\text{init}}=0.95$ as the controlling parameters.

### D.2 Baselines

In this section, we provide further clarification regarding the baselines used in our comparative analysis, beyond the backbone solvers. We independently executed LKH-3, LNS, and L2D using consistent parameters. Results for SIL were sourced from Luo et al. (2024), L2R from Zhou et al. (2025a), and all other baselines from Zheng et al. (2024). When multiple variants of a baseline were presented in the original publications, we selected the configuration that achieved the best objective values. Since the original implementation of NDS (Hottung et al., 2025) was evaluated on NVIDIA A100 GPUs whereas our experiments use NVIDIA V100 GPUs, we re-ran NDS on our hardware for fair comparison.

It is important to note that all reported results were evaluated on identical test instances (for CVRPs) or on instances sampled from the same distribution (for VRPTWs), ensuring fair comparison. Moreover, our experiments were conducted on hardware with less powerful GPUs compared to those utilized in Luo et al. (2024), Zheng et al. (2024), Zhou et al. (2025a). This hardware discrepancy suggests that the performance advantages demonstrated by our proposed model would likely persist or potentially increase if all methods were evaluated on identical computing infrastructure.

We re-implemented the backbone solvers and L2D (Li et al., 2021) to ensure a fair and strong comparison. Notably, prior studies (Zheng et al., 2024, Ye et al., 2024) did not explore configurations optimized for L2D’s full potential. Specifically, they imposed overly conservative limits (e.g., only allowing 1 trail) on LKH-3 local search updates and did not supply current solution information to the LKH-3 solver during the resolution process. This significantly weakened L2D’s performance in their benchmarks. In contrast, our comparison reflects L2D’s best achievable performance.

### D.3 Parameters and Training Hyperparameters

Parameters. Table [8](https://arxiv.org/html/2507.01037v2#A4.T8) lists the values of parameters used in training data generation and inference.

**Table 8: A list of parameters and their values used in our experiments for training and inference.**
| Training Data Generation |  |
| --- | --- |
| Parameter | Value |
| # of instances $N_{\mathcal{P}}$ | 1000 |
| # of iterative steps $T_{IS}$ | 40 |
| Improvement threshold $\eta_{\text{improv}}$ | 0 |
| Sample coefficient $\alpha_{AC}$ | 0  for small-capacity CVRPs and VRPTWs |
|  | 0.4  for large-capacity CVRPs |
| Inference |  |
| Parameter | Value |
| Threshold $\eta$ for NAR model | 0.6 |
| # of K-MEANS clusters $n_{\text{KMEANS}}$ | 3 |
| # of LKH-3 local search updates limit | 1000 |
| per iterative step |  |
| Solve time limits | 150s, 240s  for large-capacity CVRP2k, 5k |
|  | 2.5m, 4m, 5m  for small-capacity CVRP1k, 2k, 5k |
|  | 2m, 4m, 10m  for VRPTW1k, 2k, 5k |

Training Hyperparameters.
For model training, we optimize both NAR and AR architectures using the ADAM optimizer with a consistent batch size of 128 across 200 epochs for all problem variants. The learning rate is calibrated at $10^{-3}$ for large-capacity CVRPs and $10^{-4}$ for small-capacity CVRPs and VRPTWs. The loss function employs weighted components with $w_{\text{pos}}=9$, $w_{\text{insert}}=0.8$, and $w_{\text{delete}}=0.2$. All computational experiments are conducted on a single NVIDIA V100 GPU, with training duration ranging from approximately 0.5 to 1.5 days, scaling with problem dimensionality.

Regarding network architecture, our encoder maps node features $\mathbf{X}\in\mathbb{R}^{n\times 25}$ for standard problems ($\mathbf{X}\in\mathbb{R}^{n\times 28}$ for VRPTWs) to node embeddings via $\mathbf{h}^{\mathrm{init}}_{i}\!=\!\text{Concat}(\mathbf{h}^{\mathrm{MLP}}_{i},\mathbf{h}^{\mathrm{POS}}_{i})\in\mathbb{R}^{2d_{h}}$, where $d_{h}=128$. They then undergo processing through $L_{\mathrm{TFM}}=2$ Transformer layers (Vaswani, 2017) with route-specific attention masks, followed by a Graph Attention Network to derive the final node embeddings $\mathbf{H}^{\mathrm{GNN}}$. The transformer implementation utilizes 2 attention heads, 0.1 dropout regularization, ReLU activation functions, layer normalization, and feedforward dimensionality of 512. Our GNN employs a transformer convolution architecture with 2 layers ($L_{\text{GNN}}=2$) and a single attention head.

Supplementary to the specifications in Section [4](https://arxiv.org/html/2507.01037v2#S4), we delineate additional hyperparameters for our decoder modules. The NAR decoder computes $\mathbf{p}^{\mathrm{NAR}}$ (node instability probabilities) via an MLP with sigmoid activation for final probability distribution. The AR decoder incorporates single-layer Gated Recurrent Units (GRUs), complemented by a single-layer/single-head transformer for the deletion mechanism and a four-layer/single-head transformer for the insertion procedure.

All the training hyperparameters are summarized in Table [9](https://arxiv.org/html/2507.01037v2#A4.T9).

**Table 9: A list of hyperparameters and their values used in our model architecture and training.**
| Training Configuration |  |
| --- | --- |
| Parameter | Value |
| Optimizer | ADAM |
| Batch size | 128 |
| # of epochs | 200 |
| Learning rates | $10^{-3}$  for large-capacity CVRPs |
|  | $10^{-4}$  for small-capacity CVRPs and VRPTWs |
| Weight of unstable nodes $w_{\text{pos}}$ | 9 |
| Weight of prediction in insert stage $w_{\text{insert}}$ | 0.8 |
| Weight of prediction in delete stage $w_{\text{delete}}$ | 0.2 |
| Computing Resource | Single NVIDIA V100 GPU |
| Model Architecture |  |
| Parameter | Value |
| Hidden dimension | 128 |
| Encoder Transformer |  |
| # of layers $L_{\text{TFM}}$ | 2 |
| # of attention heads | 2 |
| Dropout regularization | 0.1 |
| Activation function | ReLU |
| Feedforward dimension | 512 |
| Normalization | Layer normalization |
| Encoder GNN |  |
| Architecture | Transformer Convolution Network |
| # of layers $L_{\text{GNN}}$ | 2 |
| # of attention heads | 1 |
| Decoder Components |  |
| NAR decoder activation function | Sigmoid |
| # of layers in GRUs | 1 |
| AR Transformer in Deletion Stage |  |
| # of layers $L^{\text{MHA}}_{\text{delete}}$ | 1 |
| # of attention heads | 1 |
| AR Transformer in Insertion Stage |  |
| # of layers $L^{\text{MHA}}_{\text{insert}}$ | 4 |
| # of attention heads | 1 |

### D.4 Instance Generation

In general, we generate all training and test instances following established methodologies: Zheng et al. (Zheng et al., 2024) for CVRP and Solomon (Solomon, 1987) for VRPTW. Specifically, For small-capacity CVRPs, nodes are uniformly distributed within the $[0,1]$ square, with integer demands ranging from 1 to 9 (inclusive). Vehicle capacities are set to $C=200$, $300$, and $300$ for problem sizes 1k, 2k, and 5k, respectively.
For large-capacity CVRPs, we maintain identical configurations except for increased vehicle capacities of $C=500$ and $1000$ for CVRP1k and CVRP5k, respectively.
For VRPTWs, we adopt the same spatial distribution, demand structure, and capacity constraints as the small-capacity CVRPs. Service times are uniformly set to 0.2 time units for each customer and 0 for the depot. Time windows are generated according to the methodology outlined in Solomon (Solomon, 1987).

Our experimental framework comprises distinct datasets for training, validation, and testing:

- •
Training: 1,000 instances for each problem type and scale to generate training labels
- •
Validation: 30 instances per problem configuration
- •
Testing: For small-capacity CVRPs, we utilize the 1,000 test instances from Zheng et al. (Zheng et al., 2024); for large-capacity CVRPs and VRPTWs, we evaluate on 100 instances sampled from the same distribution as the training data

## Appendix E Additional Experiments and Analysis

Figure: (a)
Refer to caption: https://arxiv.org/html/2507.01037/figures/kmeans.png

### E.1 Hyperparameter Study

Figure [11](https://arxiv.org/html/2507.01037v2#A5.F11) depicts the effects of $n_{\text{KMEANS}}$ and $\eta$. We observe that the best performance is when $n_{\text{KMEANS}}=3$ and $\eta=0.6$, suggesting that designating a moderate proportion of edges as unstable represents the most effective strategy.

### E.2 Results on Realistic Routing Datasets

We further evaluate L2Seg on the CVRPLib realistic routing dataset (Uchoa et al., 2017, Arnold et al., 2019), adhering to the settings established in Zheng et al. (2024), which incorporates instances from CVRP Set-X [54] and the very large-scale CVRP dataset Set-XXL in the test set. The instances within CVRPLib exhibit more realistic spatial distributions (distinct from simplistic uniform or clustered patterns), greater diversity, and better representation of real-world logistical challenges. For this evaluation, we employ models trained on synthetic small-capacity CVRP2k and CVRP5k datasets and zero-shot transfer them to CVRPLib. Time constraints of 240s and 600s are implemented for L2Seg during testing. Additional methodological details are provided in Appendix [D](https://arxiv.org/html/2507.01037v2#A4).
As demonstrated in Table [10](https://arxiv.org/html/2507.01037v2#A5.T10), LNS augmented with L2Seg-SYN surpasses all other learning-based methods in performance. Significantly, the computational time required by LNS+L2Seg-SYN (600s) is substantially less than that of the previously best-performing learning-based model, UDC-$\boldsymbol{x}_{250}$. These results further substantiate L2Seg’s exceptional generalizability across varied problem distributions.

**Table 10: CVRPLib results. We present the gap to the best known solutions (%).**
| Dataset, $N\in$ | LEHD | ELG aug$\times$8 | GLOP-LKH3 | TAM(LKH3) |
| --- | --- | --- | --- | --- |
| Set-X,(500,1,000] | 17.4% | 7.8% | 16.8% | 9.9% |
| Set-XXL,(1,000,10,000] | 22.2% | 15.2% | 19.1% | 20.4% |
| Dataset, $N\in$ | UDC-$\boldsymbol{x}_{2}$ | UDC-$\boldsymbol{x}_{250}$ | LNS+L2Seg-SYN (240s) | LNS+L2Seg-SYN (600s) |
| Set-X,(500,1,000] | 16.5% | 7.1% | 7.5% | 6.9% |
| Set-XXL,(1,000,10,000] | 31.3% | 13.2 % | 12.5% | 12.0% |

### E.3 Results on Clustered CVRP and Heterogeneous-demand CVRP

**Table 11: Results on clustered CVRP and heterogeneous-demand CVRP. We present gains to the backbone solver LNS and the performance of LKH-3 for reference.**
| Methods | Clustered CVRP2k | Clustered CVRP5k |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Obj.$\downarrow$ | Gain$\uparrow$ | Time$\downarrow$ | Obj.$\downarrow$ | Gain$\uparrow$ | Time$\downarrow$ |  |
| LKH-3 (Helsgaun, 2017) (for reference) | 42.06 | - | 150s | 62.33 | - | 240s |
| LNS (Shaw, 1998) | 41.54 | 0.00% | 150s | 61.42 | 0.00% | 240s |
| L2Seg-SYN-LNS (zero-shot transfer) | 41.03 | 1.23% | 150s | 60.87 | 0.90% | 240s |
| L2Seg-SYN-LNS | 40.73 | 1.95% | 150s | 60.11 | 2.13% | 240s |
| Methods | Hetero-demand CVRP2k | Hetero-demand CVRP5k |  |  |  |  |
| Obj.$\downarrow$ | Gain$\uparrow$ | Time$\downarrow$ | Obj.$\downarrow$ | Gain$\uparrow$ | Time$\downarrow$ |  |
| LKH-3 (Helsgaun, 2017) (for reference) | 46.02 | - | 150s | 65.89 | - | 240s |
| LNS (Shaw, 1998) | 45.77 | 0.00% | 150s | 64.81 | 0.00% | 240s |
| L2Seg-SYN-LNS (zero-shot transfer) | 44.35 | 3.10% | 150s | 64.28 | 0.82% | 240s |
| L2Seg-SYN-LNS | 44.15 | 3.54% | 150s | 64.15 | 1.02% | 240s |

To demonstrate L2Seg’s robustness across diverse and more realistic scenarios beyond uniform distributions, we provide in-distribution and zero‑shot generalization evaluation of our L2Seg on instances with different customer and demand distributions.

Following Li et al. (2021), we generate clustered CVRP instances with 7 clusters. For heterogeneous-demand scenarios, we employ a skewed distribution where high and low demands ($d\in\{1,2,8,9\}$) occur with probability 0.2 each, while others ($d\in\{3,4,5,6,7\}$) occur with probability 0.04 each. All experiments use LNS as the backbone solver, with LKH-3 included for reference.

Table [11](https://arxiv.org/html/2507.01037v2#A5.T11) presents the comprehensive results. L2Seg demonstrates consistent improvements across all settings: zero-shot transfer achieves 1.23% to 3.10% gains over LNS, while in-distribution testing reaches 1.02% to 3.54% improvements depending on problem size and variant. These experiments demonstrate that L2Seg maintains consistent improvements across diverse real-world conditions, from uniform spatial layouts to clustered distributions and heterogeneous demands.

### E.4 Standard Deviation Comparison

In this section, we provide standard deviation statistics for L2Seg-SYN across three different backbone solvers on large-capacity CVRPs. We conduct 5 independent trials using different random seeds for each method. All experiments are terminated at the specified time limit, and we report the standard deviations of the objective values for all 6 methods. The results are presented in Table [12](https://arxiv.org/html/2507.01037v2#A5.T12). While LKH-3 exhibits the lowest variance among baseline methods, our L2Seg approach also demonstrates consistently low variance across different problem types and backbone solvers, confirming both the effectiveness and stability of our method.

**Table 12: Performance comparison of backbone solvers with and without L2Seg-SYN on large-scale CVRP instances. Results represent mean objective values $\pm$ standard deviation across 5 independent trials of testing. L2Seg-SYN demonstrates consistent performance improvements with low variance, indicating both effectiveness and stability of the approach.**
|  | CVRP2k |  | CVRP5k |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Methods | Obj.$\downarrow$ | Gain$\uparrow$ | Time$\downarrow$ |  | Obj.$\downarrow$ | Gain$\uparrow$ | Time$\downarrow$ |
| LKH-3 Helsgaun (2017) | $45.24\pm 0.17$ | 0.00% | 152s |  | $65.34\pm 0.29$ | 0.00% | 242s |
| LKH+L2Seg-SYN | $43.92\pm 0.20$ | 2.92% | 152s |  | $64.12\pm 0.34$ | 1.87% | 248s |
| LNS Shaw (1998) | $44.92\pm 0.24$ | 0.00% | 154s |  | $64.69\pm 0.37$ | 0.00% | 246s |
| LNS+L2Seg-SYN | $43.42\pm 0.22$ | 3.34% | 152s |  | $63.94\pm 0.35$ | 1.16% | 241s |
| L2D Li et al. (2021) | $43.69\pm 0.21$ | 0.00% | 153s |  | $64.21\pm 0.32$ | 0.00% | 243s |
| L2D+L2Seg-SYN | $43.35\pm 0.23$ | 0.78% | 157s |  | $63.89\pm 0.34$ | 0.50% | 248s |

### E.5 Case Study: Comparison of Predictions of Three L2Seg Approaches

Figure: (a) L2Seg-SYN prediction
Refer to caption: https://arxiv.org/html/2507.01037/figures/case_SYN.jpg

We present a case study on a small-capacity CVRP1k instance to analyze model prediction behavior. Since the learned model ultimately predicts on two adjacent routes, we visualize unstable edge predictions (red dashed lines) for two such routes using L2Seg-SYN, L2Seg-NAR, and L2Seg-AR in Figure [12](https://arxiv.org/html/2507.01037v2#A5.F12).
L2Seg-SYN demonstrates selective prediction behavior, avoiding boundary edges while targeting specific unstable edges within route interiors—a pattern consistent with our observations in Appendix [B.1.1](https://arxiv.org/html/2507.01037v2#A2.SS1.SSS1). L2Seg-NAR successfully identifies unstable regions (route interiors) but lacks discrimination, predicting nearly all edges within these regions as unstable without capturing local dependencies. L2Seg-AR exhibits selective prediction within regions but fails to properly identify unstable regions, as many predictions occur at boundaries.
These results provide insight into L2Seg-SYN’s hybrid approach: the NAR component first identifies unstable regions, while the AR component leverages local information to make accurate predictions within each identified region.

### E.6 Unstable and Stable Edges Convergence

We conducted experiments measuring overlapping predicted edges between adjacent iterations over the first 10 rounds, revealing interesting dynamics: The overlap of predicted unstable edges increases from 28% to 54%, while stable edge overlap increases from 47% to 69% across iterations, shown in the Table [13](https://arxiv.org/html/2507.01037v2#A5.T13). This indicates gradual but not rapid convergence, allowing our method to continuously explore new regions for re-optimization rather than getting trapped in fixed segments.

**Table 13: Unstable and stable edges convergence at the first 10 iterations**
| Round # | 1 | 2 | 3 | 5 | 7 | 9 |
| --- | --- | --- | --- | --- | --- | --- |
| Unstable Edge Overlapping Percentage | 28.2% | 33.5% | 41.2% | 49.2% | 48.8% | 54.1% |
| Stable Edge Overlapping Percentage | 47.2% | 58.2% | 60.5% | 64.7% | 67.3% | 69.4% |
| Avg Segment Length | 2.45 | 2.57 | 2.44 | 3.04 | 2.87 | 2.73 |

## Appendix F Broader Impacts

On one hand, the integration of deep learning into discrete optimization offers promising advances for real-world domains such as public logistics and transportation systems, where additional considerations for social equity and environmental sustainability can be incorporated. On the other hand, the application of deep learning methodologies in discrete optimization necessitates substantial computational resources for model training, potentially leading to increased energy consumption and carbon emissions. The quantification and mitigation of these environmental impacts represent critical areas for ongoing research and responsible implementation.

## Appendix G Large Language Models Usage

We used LLMs to assist with manuscript revision. After completing the initial draft without LLM assistance, we consulted LLMs for suggestions on improving specific text passages. All LLM-generated advice was carefully reviewed to ensure accuracy before incorporation. LLMs were not used for research tasks or any purpose beyond text refinement.
