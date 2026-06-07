PHÂN TÍCH SO SÁNH CHUYÊN SÂU 8 NGHIÊN CỨU VỀ JOB SHOP SCHEDULING VỚI DEEP REINFORCEMENT LEARNING
1. TỔNG QUAN CÁC NGHIÊN CỨU
#	Paper	Tác giả chính	Venue/Năm	Bài toán	Phương pháp chính
1	JSSEnv	Pierre Tassel et al.	AAAI 2021	JSP	RL Environment + PPO (single-agent dispatcher)
2	GNN Scheduler (RL-GNN)	Junyoung Park et al.	IJPR 2021	JSP	GNN + PPO construction heuristic
3	HGNN (Heterogeneous GNN)	Wen Song et al.	IEEE TII 2023	FJSP	Heterogeneous GNN + PPO construction heuristic
4	L2S (Learning to Search)	Cong Zhang et al.	ICLR 2024	JSP	GNN-based improvement heuristic + DRL
5	Residual Scheduling	Anonymous	NeurIPS 2023 (submitted)	JSP/FJSP	Residual graph + GIN construction heuristic
6	TBGAT	Cong Zhang et al.	ICML 2024	JSP	Topology-aware Bidirectional GAT + improvement heuristic
7	RESCHED	Xiangjie Xiao et al.	ICLR 2026	FJSP/JSP/FFSP	Transformer + minimal state + REINFORCE
8	JSS Benchmark	Robbert Reijnen et al.	2023/2025	JSP/FSP/FJSP/AJSP	Benchmarking platform (không đề xuất method mới)
2. ĐIỂM GIỐNG NHAU (COMMONALITIES)
2.1 Nền tảng lý thuyết chung
Tất cả 8 papers đều:

Bài toán NP-hard: Đều nhận định JSP/FJSP là NP-hard, không thể tìm lời giải tối ưu trong thời gian hợp lý cho instances lớn.
Disjunctive Graph: Sử dụng biểu diễn bằng đồ thị phân ly (disjunctive graph) G = (O, C, D) làm nền tảng biểu diễn bài toán, trong đó O là tập operations, C là conjunctive arcs (precedence constraints), D là disjunctive arcs (machine sharing).
Mục tiêu Makespan: Đều tối ưu hóa makespan (thời gian hoàn thành toàn bộ jobs) C_max.
MDP Formulation: Mô hình hóa quá trình scheduling như Markov Decision Process (MDP) với state, action, reward, transition.
2.2 Phương pháp học chung
Deep Reinforcement Learning: Cả 7 papers nghiên cứu (trừ Benchmark) đều dùng DRL để học policy scheduling.
Graph Neural Network: 6/7 papers dùng GNN (hoặc biến thể) để encode đồ thị scheduling.
Size-agnostic: Đều hướng tới việc train trên instances nhỏ và generalize lên instances lớn.
So sánh với PDRs: Tất cả đều benchmark so với Priority Dispatching Rules (SPT, MWKR, FIFO...) và đều vượt trội hơn.
2.3 Kỹ thuật training chung
PPO chiếm ưu thế: 5/7 papers dùng Proximal Policy Optimization (PPO) làm thuật toán training chính (JSSEnv, RL-GNN, HGNN, L2S, Residual Scheduling).
Policy Gradient: Tất cả đều thuộc nhóm policy-based methods.
Reward design: Đều thiết kế reward liên quan đến makespan hoặc machine utilization.
3. ĐIỂM KHÁC NHAU CỐT LÕI
3.1 Paradigm: Construction vs. Improvement Heuristics
Đây là sự phân hóa quan trọng nhất:

Construction Heuristics	Improvement Heuristics
JSSEnv, RL-GNN, HGNN, Residual Scheduling, RESCHED	L2S, TBGAT
Xây dựng lời giải từng bước từ partial → complete solution	Bắt đầu từ complete solution, cải thiện iteratively
Action: chọn operation/machine pair để dispatch	Action: chọn cặp operations để swap trên critical path
Nhanh hơn (1 pass)	Chậm hơn nhưng chất lượng cao hơn
Khó encode WIP information vào disjunctive graph	Encode complete solution → đầy đủ thông tin
Phân tích chuyên sâu: L2S (ICLR 2024) là paper đầu tiên chỉ ra rằng paradigm construction bị hạn chế bởi việc disjunctive graph không thể encode đầy đủ partial solutions. Khi chuyển sang improvement, graph luôn biểu diễn complete solution → tận dụng tối đa cấu trúc topo. TBGAT kế thừa và nâng cao điều này bằng bidirectional topology-aware embedding.

3.2 Kiến trúc Neural Network
Paper	Architecture	Đặc điểm
JSSEnv	MLP (matrix state)	State = ma trận
RL-GNN	GIN (Graph Isomorphism Network)	Undirected message passing
HGNN	Heterogeneous GNN (2-stage)	Phân biệt operation nodes vs machine nodes
L2S	GIN + GAT dual modules	GIN cho topology, GAT cho subgraph context
Residual Scheduling	GIN	Đơn giản nhưng hiệu quả nhờ residual state
TBGAT	Bidirectional Graph Attention	Forward + backward view riêng biệt
RESCHED	Transformer (self + cross attention)	Không dùng GNN, dùng dot-product attention
Phân tích chuyên sâu về evolution kiến trúc:


GIN (undirected) → Heterogeneous GNN → Bidirectional GAT → Transformer
   RL-GNN            HGNN/L2S              TBGAT              RESCHED
   (2020-21)         (2022-23)             (2024)             (2026)
Xu hướng rõ ràng: từ GNN đơn giản xử lý undirected graphs → nhận ra directed nature quan trọng → exploit topology → bỏ hẳn GNN inductive bias để dùng Transformer linh hoạt hơn.

3.3 State Representation
Paper	Số features	Đặc điểm state
JSSEnv	Nhiều (matrix-based)	Image-like: processing time, schedule, utilization
RL-GNN	~6 features/node	Node features trên disjunctive graph
HGNN	~8-10 features	Heterogeneous: operation features + machine features + edge features
L2S	3 features (p, est, ?)	Compact, encode complete solution
Residual Scheduling	5-6 features	Loại bỏ operations đã hoàn thành (residual)
TBGAT	3 features (p, est/lst, topo_sort)	Forward: (p, EST, Φ→), Backward: (p, LST, Φ←)
RESCHED	4 features	Operation AT, Machine AT, Duration, Min Duration
Phân tích chuyên sâu: RESCHED (ICLR 2026) là đỉnh cao của xu hướng minimalist. Paper chứng minh rằng:

Hơn 20 handcrafted features trong các papers trước là redundant
Historical information trong state gây hại cho learning
Chỉ 4 features đủ để satisfy Markov property thông qua subproblem reformulation
Residual Scheduling (NeurIPS 2023) là tiền thân của ý tưởng này — loại bỏ operations đã xong để giảm noise.

3.4 Graph Structure & Topology Awareness
Paper	Xử lý topology	Graph type
RL-GNN	Undirected (bỏ qua directions)	Static disjunctive graph
HGNN	Heterogeneous (op + machine nodes)	Augmented with machine nodes
L2S	Separate GIN + GAT	Directed but without explicit topo features
TBGAT	Explicit forward/backward topo sort	Bidirectional DAG
RESCHED	RoPE cho intra-job ordering	Backward-looking O2O + O2M edges
Residual Scheduling	Shrinking graph (remove finished)	Dynamic residual graph
Phân tích chuyên sâu: TBGAT đưa ra insight quan trọng nhất về topology:

Disjunctive graphs là directed (DAG khi encode solution) → GNN cho undirected graphs (như GIN) mất thông tin
Topological sort ↔ bijection với solution space
Forward topo sort tương ứng EST (earliest starting time)
Backward topo sort tương ứng LST (latest starting time)
Đề xuất MPTS (Message Passing Topological Sort) để tính GPU-parallel
3.5 Bài toán mục tiêu
Paper	JSP	FJSP	FSP	FFSP	AJSP	SDST	Dynamic
JSSEnv	✓						
RL-GNN	✓						
L2S	✓						
TBGAT	✓						
HGNN		✓					
Residual Scheduling	✓	✓					
RESCHED	✓	✓		✓			
JSS Benchmark	✓	✓	✓		✓	✓	✓
Phân tích: RESCHED là paper đầu tiên đạt generalization across variants (FJSP → JSSP → FFSP) với cùng một framework. JSS Benchmark bao phủ nhiều nhất nhưng không đề xuất method mới.

4. SO SÁNH HIỆU NĂNG & KẾT QUẢ
4.1 Chất lượng lời giải (Optimality Gap)
Xếp hạng theo thứ tự performance trên classic benchmarks (từ tốt → kém):

TBGAT — SOTA trên improvement heuristics, gap nhỏ nhất (~1-3% trên large instances)
L2S — Improvement heuristic, vượt Or-tools CP-SAT trên large instances
RESCHED — SOTA construction heuristic cho FJSP, competitive trên JSSP
Residual Scheduling — 49/50 zero-gap trên instances >150 jobs × 20 machines (ấn tượng)
HGNN — Tốt cho FJSP, vượt PDRs đáng kể
RL-GNN — Vượt PDRs nhưng gap còn lớn
JSSEnv — Gần state-of-the-art COP nhưng instance-specific training
4.2 Computational Efficiency
Paper	Complexity	Inference Speed
JSSEnv	Instance-specific (online training)	Chậm (phải train mỗi instance)
RL-GNN	O(	O
HGNN	O(	O
L2S	O(	O
TBGAT	**O(	J
Residual Scheduling	Giảm dần theo thời gian	Nhanh (graph shrinks)
RESCHED	O(n² + n×m) attention	Nhanh (single pass, parallelizable)
Điểm nổi bật:

TBGAT chứng minh toán học linear complexity O(|J| + |M|) per layer — rất quan trọng cho scalability.
Residual Scheduling tự động giảm computation vì graph nhỏ dần khi operations hoàn thành.
4.3 Generalization Capability
Paper	Train size → Test size	Cross-distribution	Cross-variant
JSSEnv	Không generalize (online)	✗	✗
RL-GNN	6×6 → 10×10, 15×15	Có	✗
HGNN	10×5 → 20×5, 20×10	Có	✗
L2S	15×15 → 20×20, 30×20	Có	✗
Residual Scheduling	Small → 150×20 (zero-gap!)	Rất tốt	JSP↔FJSP
TBGAT	10×10 → 50×20, 100×20	Rất tốt	✗
RESCHED	10×10 → 50×20, 100×20	Rất tốt	FJSP↔JSSP↔FFSP
5. PHÂN TÍCH CHUYÊN SÂU THEO CHỦ ĐỀ
5.1 Evolution của Reward Design
Paper	Reward Type	Công thức
JSSEnv	Dense (machine utilization)	Phản ánh tác động allocation lên utilization
RL-GNN	Sparse (negative makespan)	r = -C_max tại cuối episode
HGNN	Incremental makespan	r_t = LB(s_t) - LB(s_{t+1})
L2S	Step-wise improvement	r = max(C_max(s*) - C_max(s_{t+1}), 0)
Residual Scheduling	Makespan-based	Standard makespan reward
TBGAT	Same as L2S	Cumulative = improvement from initial
RESCHED	Lower-bound difference	r_t = -(FT_max(s_{t+1}) - FT_max(s_t))
Insight: JSSEnv là paper đầu tiên chỉ ra dense reward (thay vì sparse makespan) giúp training ổn định hơn. RESCHED tinh chỉnh thêm bằng estimated lower-bound.

5.2 Xử lý Flexibility (FJSP)
Flexibility (1 operation → nhiều machines) đặt ra 2 thách thức:

Decision complexity: Phải chọn cả operation VÀ machine
Representation: Quan hệ one-to-many operation↔machine
Paper	Cách xử lý
HGNN	Composite decision (op, machine) pair; heterogeneous graph with machine nodes
Residual Scheduling	O→M và M→O edges; average processing time cho features
RESCHED	O2M connection as graph structure; Duration as edge feature; cross-attention
HGNN đặt nền móng cho cách tiếp cận heterogeneous graph — chia node thành 2 loại (operation, machine) với edges khác nhau. RESCHED kế thừa nhưng đơn giản hóa bằng cách encode Duration vào edge attention thay vì node features.

5.3 Critical Path và Neighborhood Structure
Improvement heuristics (L2S, TBGAT) phụ thuộc vào N5 neighborhood structure:

Tìm critical path trên disjunctive graph
Xác định critical blocks (nhóm operations liên tiếp trên cùng machine trên critical path)
Swap first/last pair trong mỗi block
L2S dùng GNN học policy để chọn cặp swap (thay vì evaluate toàn bộ neighborhood).
TBGAT nâng cấp bằng:

Forward/backward topology encoding giúp phân biệt chất lượng solutions
Attention mechanism cho phép adaptive weighting của neighbors
5.4 Vai trò của JSS Benchmark Platform
Paper JSS Benchmark (Reijnen et al.) khác biệt hoàn toàn — đây là infrastructure paper:

Cung cấp unified implementation cho JSP, FSP, FJSP, AJSP + SDST + dynamic variants
Tích hợp cả heuristics, metaheuristics, exact methods, và DRL
Mục đích: standardized comparison, reproducibility
Bao gồm implementations của nhiều methods (L2D, RL-GNN, HGNN...)
Đây là meta-paper kết nối tất cả các nghiên cứu còn lại vào cùng 1 framework đánh giá.

6. DÒNG THỜI GIAN PHÁT TRIỂN (RESEARCH TRAJECTORY)

2021: JSSEnv (RL environment, online training, matrix state)
  │     RL-GNN (GNN + disjunctive graph, construction, transferable)
  │
  ▼
2022-23: HGNN (mở rộng sang FJSP, heterogeneous graph)
  │       Residual Scheduling (loại bỏ irrelevant info, shrinking graph)
  │
  ▼
2024: L2S (chuyển sang improvement paradigm, GNN dual-module)
  │    TBGAT (topology-aware, bidirectional, MPTS algorithm)
  │
  ▼
2025-26: RESCHED (minimalist state, Transformer thay GNN, cross-variant)
         JSS Benchmark (unified platform)
Xu hướng rõ ràng:

Simplification of state (20+ features → 4 features)
Architecture shift (GIN → Het-GNN → GAT → Transformer)
Topology exploitation (implicit → explicit forward/backward)
Problem generalization (JSP-only → FJSP → multi-variant)
Paradigm diversification (construction-only → improvement → hybrid)
7. STRENGTHS & LIMITATIONS CỦA TỪNG PAPER
Paper	Strengths	Limitations
JSSEnv	Dense reward design; handles stochastic; search-space reduction	Không size-agnostic; online training chậm; không generalize
RL-GNN	Đầu tiên dùng GNN cho JSSP; transferable policy; elegant formulation	Static graph; undirected → mất thông tin topology; gap còn lớn
HGNN	Xử lý flexibility tốt; heterogeneous design; efficient	Chỉ FJSP; complex feature engineering; PDR-level performance gap
L2S	Breakthrough sang improvement; outperform CP-SAT on large	Chỉ JSP; GIN cho directed graph chưa optimal; 2 module riêng biệt
Residual Scheduling	Ý tưởng đẹp; zero-gap impressive; đơn giản	GIN architecture cũ; thiếu topology awareness; chưa peer-reviewed
TBGAT	Topology-aware; linear complexity proven; new SOTA	Chỉ JSP; improvement cần initial solution; MPTS overhead
RESCHED	Minimalist; multi-variant; Transformer scalable; SOTA FJSP	REINFORCE high variance; chưa improvement paradigm; edge attention cost
JSS Benchmark	Comprehensive; reproducible; community resource	Không contribute method mới; DRL implementations có thể outdated
8. KẾT LUẬN & NHẬN ĐỊNH
8.1 Các đóng góp then chốt theo thứ tự impact
RL-GNN (Park 2021): Mở đường cho GNN-based construction heuristics — seminal work.
L2S (Zhang, ICLR 2024): Paradigm shift từ construction → improvement, chứng minh disjunctive graph encode complete solutions tốt hơn partial.
RESCHED (Xiao, ICLR 2026): Chứng minh minimalist state đủ (Markov sufficiency), Transformer thay thế GNN thành công.
TBGAT (Zhang 2024): Lý thuyết hóa vai trò topology trong JSSP, đề xuất MPTS.
HGNN (Song, IEEE TII 2023): Đặt nền cho FJSP với heterogeneous graph.
Residual Scheduling: Ý tưởng "loại bỏ irrelevant" đơn giản nhưng hiệu quả đáng kinh ngạc.
8.2 Gap nghiên cứu còn tồn tại
Dynamic scheduling: Chỉ JSS Benchmark đề cập, chưa có DRL method mạnh cho online arrivals.
Multi-objective: Tất cả chỉ optimize makespan; thực tế cần tardiness, energy, cost...
Improvement + Construction hybrid: Chưa có paper kết hợp tốt cả 2 paradigms.
Real-world constraints: SDST, machine breakdowns, buffer limits chưa được DRL xử lý tốt.
Scalability >1000 operations: Chưa paper nào demonstrate thuyết phục trên industrial-scale.
8.3 Hướng phát triển tương lai dự đoán
Dựa trên trajectory: Transformer-based improvement heuristic với minimal state, multi-variant, và sampling-based decoding là hướng đi tiếp theo hợp lý — kết hợp strengths của RESCHED (architecture + state) với paradigm của L2S/TBGAT (improvement).

Phân tích được tổng hợp từ 8 nghiên cứu trong folder "Compare Research Paper" — tất cả đều thuộc lĩnh vực Deep Reinforcement Learning for Job Shop Scheduling.