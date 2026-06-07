PHÂN TÍCH CHUYÊN SÂU: FRAMEWORK L2S VÀ TÍNH KẾ THỪA CỦA TBGAT
Phần A — L2S: Một framework, không chỉ một model
Điều quan trọng nhất cần làm rõ: L2S không phải là một kiến trúc mạng, mà là một framework học cải thiện (learning-to-search) cho JSSP. Cùng nhóm tác giả (Cong Zhang, Zhiguang Cao, Wen Song, Yaoxin Wu) sau đó tạo ra TBGAT như một thực thể mới đặt vào đúng cái khung đó. Hiểu được ranh giới "khung" vs "model" là chìa khóa để thấy TBGAT kế thừa cái gì và thay đổi cái gì.

A.1 Luận điểm nền tảng của L2S (cái mà TBGAT thừa hưởng nguyên vẹn)
L2S mở đầu bằng một phê phán paradigm, không phải một thủ thuật kỹ thuật. Lập luận cốt lõi (trang 1-2 của paper):

Construction heuristics encode partial solutions bằng disjunctive graph. Nhưng disjunctive graph không thể mang đủ thông tin work-in-progress (WIP) — machine load hiện tại, job status — vì các thông tin này tự nhiên thuộc về operations chứ không có node nào trong graph chứa được. Hệ quả: với partial solution, các disjunctive arc giữa operation chưa dispatch bị bỏ, hướng của arc trong machine clique chưa xác định → biểu diễn bị bias nghiêm trọng.

Từ đó L2S đặt câu hỏi định hướng cả dòng nghiên cứu:

"Can we transform the learning-to-schedule problem into a learning-to-search-graph-structures problem?"

Câu trả lời: chuyển sang improvement heuristic. Khi đó mỗi neighbour là một complete solution, biểu diễn được bằng một disjunctive graph DAG đầy đủ thông tin. Bài toán scheduling trở thành bài toán tìm kiếm trên không gian cấu trúc đồ thị.

Đây chính là phát biểu mà TBGAT trích dẫn nguyên văn trong phần Related Work: "L2S significantly narrows the optimality gaps by learning neural improvement heuristics for JSSP, thereby transforming the scheduling problem into a graph structure search problem."

A.2 Bộ khung 5 thành phần của L2S
Framework L2S định nghĩa 5 trụ cột. TBGAT giữ lại trụ cột 1, 2, 4 gần như nguyên vẹn, và chỉ thay trụ cột 3:

#	Thành phần	Định nghĩa trong L2S	TBGAT
1	Local search loop	Initial solution (dispatching rule) → biểu diễn disjunctive graph → policy chọn cặp swap → cập nhật → lặp đến horizon T	Giữ nguyên
2	N₅ neighborhood	Critical block trên critical path; swap cặp đầu/cuối mỗi block; |N₅(s)| ≤ 2N(s)−2	Giữ nguyên
3	Graph embedding	GIN (TPM) + GAT (CAM), hai module độc lập	Thay bằng Bidirectional GAT
4	MDP formulation	State = complete solution graph; Action = cặp operation; Reward = step-wise improvement	Giữ nguyên
5	Message-passing evaluator	Tính EST/LST song song trên GPU thay cho CPM tuần tự	Mở rộng thành MPTS
A.3 MDP của L2S — chi tiết quyết định mọi thứ phía sau
State: mỗi state là complete solution tại bước t, biểu diễn bằng disjunctive graph. Node feature là vector 3 chiều:
$$x_{ji} = (p_{ji}, est_{ji}, lst_{ji}) \in \mathbb{R}^3$$
L2S nêu một quan sát tinh tế: nếu operation nằm trên critical path thì $est_{ji} = lst_{ji}$. Vậy chỉ riêng cặp (est, lst) đã ngầm mã hoá việc node có nằm trên critical path hay không — tín hiệu cực kỳ quan trọng cho policy.

Action: chọn một cặp $(O_{ji}, O_{kl})$ trong tập candidate do N₅ sinh ra. Action space động theo từng state.

Reward (công thức 1 của L2S):
$$r(s_t, a_t) = \max(C_{max}(s^t) - C{max}(s_{t+1}), 0)$$
trong đó $s^t$ là incumbent (best-so-far). Điểm đẹp của thiết kế: cumulative reward $R_t = C{max}(s_0) - C_{max}(s^*_t)$ = đúng bằng tổng cải thiện so với lời giải ban đầu. Reward này còn xử lý được "absorbing state" — khi chỉ còn 1 critical block, không còn action khả thi, reward = 0 vĩnh viễn.

Training: L2S thiết kế n-step REINFORCE để giải hai vấn đề: (1) reward thưa khi episode dài, (2) out-of-memory với T lớn. Train mỗi n bước dọc trajectory thay vì chờ hết episode.

A.4 Hai đóng góp kỹ thuật riêng của L2S
(1) Kiến trúc dual-module (TPM + CAM):

TPM (Topological Embedding Module) dùng GIN — vì GIN có sức phân biệt đồ thị non-isomorphic mạnh nhất. Mục tiêu: phân biệt hai solution khác nhau về cấu trúc topo (hai solution luôn khác nhau ở thứ tự job trên ít nhất một máy).
CAM (Context-Aware Module) dùng GAT trên hai subgraph tách biệt: $G_J$ (chỉ conjunctive arc — precedence) và $G_M$ (chỉ disjunctive arc — thứ tự máy). Bắt được tính dị thể (heterogeneity) của hai loại neighbour.
Embedding cuối = concat của hai module: $h_V = (\mu_V^K : \nu_V^K)$.
(2) Message-passing evaluator (Theorem 4.2): thay Critical Path Method (CPM) tuần tự bằng một cơ chế truyền tin tính EST cho cả batch trên GPU. Sau tối đa H lần truyền tin ($H$ = số node trên đường dài nhất), $d_V = est_V$. LST tính bằng phiên bản backward (đảo chiều mọi arc). Bảng 2 của L2S: trên batch 512 graph, evaluator GPU nhanh hơn CPM 11.4 lần.

A.5 Kết quả định vị L2S
L2S là paper đầu tiên mà DRL vượt CP-SAT (OR-Tools) trên instance lớn — Taillard 100×20, và trên các instance cực lớn (1000 jobs, 40.000 operations) outperform CP-SAT tới −15% đến −24% gap. Đây là cột mốc: học cải thiện có thể đánh bại solver thương mại tinh vi trên quy mô công nghiệp.

Phần B — TBGAT kế thừa L2S như thế nào
TBGAT (cùng tác giả chính Cong Zhang) là sự kế thừa có chủ đích và được tuyên bố công khai. Trong paper, TBGAT mô tả framework local search của mình là "akin to L2S" và trích dẫn L2S ở mọi thành phần được giữ lại. Đây không phải kế thừa ngầm mà là một bước đi tiếp nối tường minh.

B.1 Những gì TBGAT giữ NGUYÊN từ L2S
Toàn bộ local search loop (Fig. 2 của TBGAT gần như sao chép Fig. 2 của L2S): initial solution từ dispatching rule → disjunctive graph → TBGAT chọn move → swap → lặp đến stop criterion.

N₅ neighborhood structure y hệt — critical block, swap cặp đầu/cuối, loại cặp đầu của block đầu và cặp cuối của block cuối, chọn ngẫu nhiên nếu nhiều critical path. TBGAT trích dẫn thẳng L2S và Nowicki & Smutnicki cho phần này.

MDP gần như giống hệt:

State = disjunctive graph của seed solution.
Reward trùng khít công thức 1 của L2S: $r(s_t,a_{t+1}) = \max(C_{max}(s^*) - C_{max}(s_{t+1}), 0)$, với cùng tính chất cumulative = improvement so với initial.
Action = cặp operation từ N₅; $A_t = \emptyset$ ⟺ đạt optimal.
Initial solution dùng cùng dispatching rule FDD/MWKR — TBGAT nói rõ điều này để "đảm bảo so sánh công bằng với L2S".

Cùng kế hoạch huấn luyện và đánh giá: n-step REINFORCE, train ở T=500 nhưng test ở 500/1000/2000/5000 bước, cùng 5 training size (10×10 … 20×15), cùng 7 benchmark cổ điển, p chia 99, est/lst chia 1000, cùng implement bằng PyTorch-Geometric.

Nói cách khác: TBGAT đặt một bộ não mới vào đúng bộ xương L2S, để mọi so sánh đều là so sánh "apples-to-apples" về phần representation.

B.2 TBGAT NÂNG CẤP gì — và vì sao
TBGAT mở bài bằng việc phê phán chính L2S (rất thẳng thắn). Hai điểm yếu của L2S mà TBGAT nêu đích danh (trang 4):

Vấn đề 1: "It is unclear whether GIN can maintain the same discriminative power for directed graphs as for undirected graphs." — GIN trong TPM vốn thiết kế cho đồ thị vô hướng, nhưng disjunctive graph (khi mã hoá complete solution) là DAG có hướng. Dùng GIN ở đây có thể mất thông tin hướng.

Vấn đề 2: "The GAT network cannot allocate distinct attention scores... since each node possesses only a single neighbour in either context subgraph, thus rendering the attention mechanism ineffective." — Trong CAM, mỗi node trong $G_J$ hoặc $G_M$ thường chỉ có một neighbour (job predecessor / machine predecessor), nên cơ chế attention của GAT trở nên vô nghĩa (chẳng có gì để "cân" giữa các neighbour).

Đây là phê phán sắc bén: hai module mạnh nhất của L2S đều bị dùng sai bối cảnh. TBGAT giải quyết bằng ba cải tiến mắt xích:

(1) Bidirectional view thay cho dual-module. Thay vì tách theo loại arc (conjunctive vs disjunctive như L2S), TBGAT tách theo chiều truyền tin:

FEM (Forward Embedding Module): truyền từ gốc → lá, theo chiều arc thuận, mang historical context. Raw feature: $\vec{h}^0_x = (p_x, est_x, \vec{\Phi}_x)$.
BEM (Backward Embedding Module): truyền ngược lá → gốc, mang future schedule information. Raw feature: $\overleftarrow{h}^0_x = (p_x, lst_x, \overleftarrow{\Phi}_x)$.
Quan trọng: bây giờ mỗi node có nhiều neighbour theo từng chiều → attention thực sự có việc để làm, sửa đúng Vấn đề 2. Và việc xử lý có hướng (forward/backward riêng) sửa đúng Vấn đề 1.

(2) Topological sort như feature tường minh — đóng góp lý thuyết cốt lõi. Đây là điểm TBGAT đi xa hơn L2S về mặt insight. L2S chỉ dùng (p, est, lst). TBGAT bổ sung topological sort $\vec{\Phi}, \overleftarrow{\Phi}$ làm feature, với cơ sở lý thuyết:

Lemma 1: nếu $O_{ji}$ là tiền nhiệm của $O_{mk}$ thì $\vec{\Phi}(O_{ji}) < \vec{\Phi}(O_{mk})$ và $EST_{ji} < EST_{mk}$ → forward topo sort ↔ global processing order.
Corollary 1: đối xứng cho backward sort ↔ LST.
Tồn tại song ánh (bijection) giữa không gian topo của disjunctive graph và không gian schedule khả thi → học topo = học phân biệt chất lượng solution. Đây là lý thuyết hoá điều mà L2S chỉ dùng ngầm qua "est=lst trên critical path".
(3) MPTS (Message-Passing Topological Sort) — kế thừa trực tiếp evaluator của L2S. Topo sort truyền thống tính tuần tự, không batch được trên GPU → cản trở training DRL. TBGAT tổng quát hoá đúng ý tưởng message-passing evaluator của L2S (vốn để tính EST/LST) sang tính topo sort:

Theorem 1: áp dụng toán tử MPO (max-pooling trên neighbour) $L_x$ lần thì xác định được topo sort, batch được trên GPU.
MPTS áp dụng cho mọi DAG, dùng cho cả forward $\vec{\Phi}$ và backward $\overleftarrow{\Phi}$.
TBGAT nói rõ Theorem 2 (linear complexity) được chứng minh "by following the proof of Theorem 4.1 in L2S" — kế thừa cả bộ khung chứng minh độ phức tạp tuyến tính O(|J|+|M|).

(4) Entropy-regularized REINFORCE. Cải tiến nhỏ trên n-step REINFORCE của L2S: thêm số hạng entropy $H(\pi_\theta)$ vào loss để khuyến khích exploration. Algorithm 1 của TBGAT giống hệt Algorithm 1 của L2S, chỉ khác đúng dòng 11 (thêm entropy).

B.3 Bảng tổng kết kế thừa
Thành phần	L2S (gốc)	TBGAT (kế thừa + nâng cấp)	Bản chất thay đổi
Paradigm	Improvement / learning-to-search	Giữ nguyên	Kế thừa 100%
Local search loop	Có	Giữ nguyên	Kế thừa 100%
N₅ neighborhood	Có	Giữ nguyên	Kế thừa 100%
Reward	max-improvement vs incumbent	Giữ nguyên công thức	Kế thừa 100%
Node features	(p, est, lst)	(p, est, Φ→) + (p, lst, Φ←)	Thêm topo sort
Architecture	GIN (TPM) + GAT (CAM), tách theo loại arc	Bidirectional GAT (FEM+BEM), tách theo chiều	Tái thiết kế
Evaluator	Message-passing cho EST/LST	MPTS — mở rộng cho topo sort	Tổng quát hoá
Training	n-step REINFORCE	+ entropy regularization	Tinh chỉnh
Complexity proof	Theorem 4.1: O(|J|+|M|)	Theorem 2: "follows L2S proof"	Kế thừa khung CM
B.4 Bằng chứng thực nghiệm cho tính kế thừa-vượt-trội
Vì TBGAT giữ nguyên mọi thứ trừ representation, chênh lệch hiệu năng cô lập đúng đóng góp của bidirectional + topology:

TBGAT-500 đánh bại L2S-500 ở gần như mọi instance. Ví dụ Taillard 15×15: 8.0% vs 9.3%; 100×20: 6.4% vs 7.9%; FT 10×10: 5.2% vs 9.9%.
TBGAT-1000 đạt gap nhỏ hơn L2S-2000 trên mọi size synthetic (Table 3 của TBGAT) → hiệu quả hơn ~2× về số bước.
Sampling efficiency: Fig. 6(b) cho thấy TBGAT học nhanh hơn L2S rõ rệt — nhờ embedding module hợp lý hơn và entropy regularization.
Vượt CP-SAT mạnh hơn L2S: trên Taillard 100×20, TBGAT đạt gap −69.2% so với CP-SAT trong 6.7 phút (L2S cũng vượt nhưng biên độ nhỏ hơn).
TBGAT tìm được optimal cho FT 6×6, LA 15×5, LA 20×5, LA 30×10 — "while L2S fails to do so" (trích nguyên văn).
Phần C — Nhận định: bản chất mối quan hệ L2S → TBGAT
Đây là quan hệ "framework → instantiation tốt hơn", không phải hai phương pháp cạnh tranh. L2S đóng góp paradigm shift (construction → improvement, scheduling → graph search) và bộ khung MDP + evaluator. TBGAT chấp nhận trọn vẹn khung đó rồi giải quyết đúng các điểm yếu kỹ thuật mà chính L2S để lại ở tầng representation.

TBGAT là một "ablation study được nâng cấp thành paper riêng". Vì controlled-comparison gần như hoàn hảo (cùng loop, cùng N₅, cùng reward, cùng initial solution, cùng training regime), TBGAT chứng minh được rằng toàn bộ phần tăng hiệu năng đến từ việc xử lý topology có hướng đúng cách — chứ không phải từ may mắn về hyperparameter hay loop.

Trục tiến hoá tư tưởng: L2S nhận ra "disjunctive graph của complete solution là DAG có hướng giàu thông tin" nhưng vẫn dùng công cụ của đồ thị vô hướng (GIN) và attention đặt sai chỗ. TBGAT đẩy logic đó đến cùng: nếu graph có hướng và topo sort có song ánh với schedule, thì hãy embed theo hướng và đưa topo sort vào làm feature. Đó là sự hoàn thiện một ý tưởng còn dang dở, không phải một hướng đi mới.

Hạn chế kế thừa: TBGAT thừa hưởng cả giới hạn của L2S — chỉ làm JSP thuần (không FJSP/FFSP), vẫn cần initial solution và phụ thuộc chất lượng N₅, và chi phí MPTS là overhead mới mà L2S không có. Đây cũng là lý do RESCHED (ICLR 2026, vẫn nhóm Cong Zhang/Zhiguang Cao) sau này rẽ sang Transformer + minimal state + đa biến thể — bước ra khỏi chính cái khung improvement này.

Tóm lại: L2S định nghĩa luật chơi; TBGAT chơi cùng luật đó nhưng với một bộ biểu diễn topo có cơ sở lý thuyết vững hơn, và thắng một cách có kiểm soát. Tính kế thừa ở đây là tường minh, có chủ đích, và là minh hoạ mẫu mực cho cách một dòng nghiên cứu tự hoàn thiện qua các paper liên tiếp của cùng nhóm tác giả.