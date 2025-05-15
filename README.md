# Ứng dụng Giải Câu đố 8 Ô Chữ (8-Puzzle Solver) và Trực Quan Hóa Thuật Toán Tìm kiếm AI

Đây là một ứng dụng GUI (Giao diện người dùng) được xây dựng bằng Tkinter trong Python, cho phép người dùng trực quan hóa quá trình giải câu đố 8 ô chữ (8-Puzzle) bằng nhiều thuật toán tìm kiếm Trí tuệ Nhân tạo (AI) khác nhau.

## Mục tiêu

Mục tiêu chính của dự án là:

1.  **Trực quan hóa** cách các thuật toán tìm kiếm khác nhau hoạt động để giải cùng một bài toán (8-Puzzle).
2.  Giúp người học AI **hiểu rõ hơn** các khái niệm về không gian trạng thái, không gian niềm tin, hàm heuristic và cách các thuật toán duyệt qua không gian đó.
3.  Cung cấp một công cụ để **so sánh hiệu suất** (thời gian, số nút/trạng thái đã duyệt, độ dài đường đi) của các thuật toán trên các bài toán cụ thể.

## Nội dung

Dự án bao gồm:

*   Cài đặt câu đố 8 ô chữ và các thao tác cơ bản (tìm ô trống, lấy hàng xóm).
*   Triển khai nhiều thuật toán tìm kiếm AI từ các nhóm khác nhau.
*   Giao diện người dùng Tkinter để hiển thị trạng thái ban đầu, trạng thái đích, trạng thái hiện tại trong quá trình giải và các thông tin hiệu suất.
*   Chức năng hoạt họa (animation) quá trình giải theo đường đi tìm được bởi thuật toán.
*   Chức năng sinh trạng thái bắt đầu mới có thể giải được (Solvable Start State Generation).

### 2.1. Các thuật toán Tìm kiếm không có thông tin (Uninformed Search)

Nhóm thuật toán này tìm kiếm lời giải mà **không sử dụng bất kỳ thông tin bổ sung nào** về "khoảng cách" hoặc "chi phí ước lượng" từ trạng thái hiện tại đến trạng thái đích. Chúng chỉ dựa vào cấu trúc của không gian tìm kiếm (các trạng thái và các kết nối giữa chúng).

*   **Các thành phần chính của bài toán tìm kiếm:**
    *   **Trạng thái (State):** Một cấu hình cụ thể của bàn cờ 8 ô chữ.
    *   **Trạng thái bắt đầu (Initial State):** [[2, 6, 5], [0, 8, 7], [4, 3, 1]].
    *   **Trạng thái đích (Goal State):** [[1, 2, 3], [4, 5, 6], [7, 8, 0]].
    *   **Các hành động/Phép toán (Actions/Operators):** Các di chuyển hợp lệ của ô trống (lên, xuống, trái, phải).
    *   **Mô hình chuyển đổi (Transition Model):** Mô tả trạng thái mới sẽ như thế nào sau khi thực hiện một hành động từ trạng thái hiện tại.
    *   **Chi phí đường đi (Path Cost):** Chi phí tích lũy của chuỗi hành động từ trạng thái bắt đầu đến trạng thái hiện tại (thường là 1 cho mỗi bước di chuyển ô trống trong 8-Puzzle).
    *   **Lời giải (Solution):** Một chuỗi các hành động từ trạng thái bắt đầu đến trạng thái đích. Một lời giải tối ưu là lời giải có tổng chi phí đường đi thấp nhất.

*   **Các thuật toán trong nhóm này được triển khai:**
    *   **BFS (Breadth-First Search):** Tìm kiếm theo chiều rộng. Duyệt qua không gian trạng thái theo từng lớp độ sâu. Đảm bảo tính đầy đủ và tối ưu trên đồ thị này.
    ![](Gif/BFS.gif)
    *   **DFS (Depth-First Search):** Tìm kiếm theo chiều sâu. Duyệt xuống sâu nhất có thể trước khi quay lui. Triển khai có sử dụng `visited` set để tránh lặp vô hạn.
    *   **(số bước giải rất lớn, thời gian giải lâu nên phần này em không có gif)
    *   **UCS (Uniform Cost Search):** Tìm kiếm chi phí đồng nhất. Mở rộng nút có chi phí đường đi thấp nhất. Đảm bảo tính đầy đủ và tối ưu.
    ![](Gif/UCS.gif)
    *   **IDS (Iterative Deepening Search):** Tìm kiếm tăng dần độ sâu. Thực hiện DLS (Depth-Limited Search) lặp đi lặp lại với giới hạn độ sâu tăng dần. Triển khai theo cách lặp tường minh để tránh giới hạn đệ quy Python. Đảm bảo tính đầy đủ và tối ưu.
    ![](Gif/IDS.gif)

*   **Hình ảnh so sánh hiệu suất:**
    *   **BFS (Breadth-First Search):
    ![](hieuSuat/BFS.png)

    *   **DFS (Depth-First Search):
    ![](hieuSuat/DFS.png)

    *   **UCS (Uniform Cost Search)
    ![](hieuSuat/UCS.png)

    *   **IDS (Iterative Deepening Search)
    ![](hieuSuat/IDS.png)


*   **Nhận xét về hiệu suất trong nhóm này khi áp dụng lên 8 ô chữ:**
    *   Về Độ dài đường đi (Path Steps):
        *   BFS, UCS, và IDS tìm thấy đường đi có độ dài 23 bước.
        *   DFS tìm thấy đường đi có độ dài 7113 bước.
    *   Về Số nút đã duyệt (Nodes):
        *   DFS duyệt ít nút nhất (7297 nút).
        *   UCS duyệt ít nút hơn BFS (103936 so với 115372 nút).
        *   IDS duyệt nhiều nút nhất (659337 nút).
    *   Về Thời gian thực thi (Time):
        *   DFS là thuật toán nhanh nhất (0.604 giây).
        *   BFS (1.297 giây) nhanh hơn UCS (1.880 giây).
        *   IDS là thuật toán chậm nhất (6.926 giây).
    *   Nhận xét tổng hợp: BFS, UCS và IDS tìm thấy đường đi tối ưu (23 bước). DFS tìm đường đi không tối ưu (7113 bước). DFS là nhanh nhất và duyệt ít nút nhất trong trường hợp này, trong khi IDS là chậm nhất và duyệt nhiều nút nhất.

### 2.2. Các thuật toán Tìm kiếm có thông tin (Informed Search)

Nhóm thuật toán này sử dụng **thông tin bổ sung** (thường là hàm heuristic - hàm ước lượng chi phí từ trạng thái hiện tại đến đích) để hướng dẫn quá trình tìm kiếm, nhằm tìm lời giải hiệu quả hơn so với các thuật toán không có thông tin.


*   **Các thành phần chính của bài toán tìm kiếm:**
    *   **Trạng thái (State):** Một cấu hình cụ thể của bàn cờ 8 ô chữ.
    *   **Trạng thái bắt đầu (Initial State):** [[2, 6, 5], [0, 8, 7], [4, 3, 1]].
    *   **Trạng thái đích (Goal State):** [[1, 2, 3], [4, 5, 6], [7, 8, 0]].
    *   **Các hành động/Phép toán (Actions/Operators):** Các di chuyển hợp lệ của ô trống (lên, xuống, trái, phải).
    *   **Mô hình chuyển đổi (Transition Model):** Mô tả trạng thái mới sẽ như thế nào sau khi thực hiện một hành động từ trạng thái hiện tại.
    *   **Chi phí đường đi (Path Cost):** Chi phí tích lũy của chuỗi hành động từ trạng thái bắt đầu đến trạng thái hiện tại (thường là 1 cho mỗi bước di chuyển ô trống trong 8-Puzzle).
    *   **Lời giải (Solution):** Một chuỗi các hành động từ trạng thái bắt đầu đến trạng thái đích. Một lời giải tối ưu là lời giải có tổng chi phí đường đi thấp nhất.
    *   **Hàm Heuristic được sử dụng:** Khoảng cách Manhattan (Manhattan Distance) - Tính tổng khoảng cách (số bước ngang + dọc) từ vị trí hiện tại của mỗi ô (không tính ô 0) đến vị trí đích của nó. Đây là một heuristic chấp nhận được (admissible) và nhất quán (consistent) cho 8-Puzzle.

*   **Các thuật toán trong nhóm này được triển khai:**
    *   **A\* (A-Star Search):** Mở rộng nút có chi phí ước lượng thấp nhất (`f(n) = g(n) + h(n)`). Với heuristic Manhattan, A\* là đầy đủ và tối ưu.
    ![](Gif/A_star.gif)
    *   **Greedy Best-First Search:** Mở rộng nút có chi phí ước lượng đến đích thấp nhất (`h(n)`). Thường nhanh nhưng không đảm bảo tính tối ưu hoặc đầy đủ.
    ![](Gif/Greedy.gif)
    *   **IDA\* (Iterative Deepening A\*):** Kết hợp tăng dần giới hạn (như IDS) với hàm heuristic (như A\*). Hiệu quả về bộ nhớ và đảm bảo tính đầy đủ/tối ưu với heuristic Manhattan.
    ![](Gif/IDA_star.gif)

*   **Hình ảnh so sánh hiệu suất:**
    *   **A\* (A-Star Search):
    ![](hieuSuat/A_star.png)

    *   **Greedy Best-First Search:
    ![](hieuSuat/Greedy.png)

    *   IDA\* (Iterative Deepening A\*):
    ![](hieuSuat/IDA_star.png)

   
*   **Nhận xét về hiệu suất trong nhóm này khi áp dụng lên 8 ô chữ:**
    *   Về Độ dài đường đi (Path Steps):
        *   A* và IDA* tìm thấy đường đi có độ dài 23 bước. Đây là độ dài tối ưu cho bài toán này.
        *   Greedy Best-First Search tìm đường đi có độ dài 79 bước.
    *   Về Số nút đã duyệt (Nodes):
        *   Greedy Best-First Search duyệt ít nút nhất (455 nút).
        *   A* duyệt số nút trung bình (1023 nút).
        *   IDA* duyệt nhiều nút nhất (3209 nút).
    *   Về Thời gian thực thi (Time):
        *   Greedy Best-First Search là nhanh nhất (0.007s).
        *   A* là nhanh thứ hai (0.013s).
        *   IDA* chậm hơn A* một chút (0.023s).
    *   Nhận xét tổng hợp:
        *   A* và IDA* tìm thấy lời giải tối ưu. Greedy Best-First Search không đảm bảo tính tối ưu.
        *   Greedy Best-First Search hiệu quả nhất về thời gian và số nút duyệt trong trường hợp này.

### 2.3. Các thuật toán Tìm kiếm cục bộ (Local Search)

Các thuật toán này thường chỉ duy trì một hoặc một vài trạng thái hiện tại và di chuyển đến các trạng thái lân cận dựa trên một tiêu chí (thường là cải thiện giá trị mục tiêu/heuristic). Chúng không ghi nhớ đường đi đầy đủ từ điểm bắt đầu và có thể mắc kẹt ở cực tiểu cục bộ hoặc không tìm thấy lời giải.

*   **Các thành phần chính của bài toán tìm kiếm:**
    *   **Trạng thái (State):** Một cấu hình cụ thể của bàn cờ 8 ô chữ.
    *   **Trạng thái bắt đầu (Initial State):** [[0, 1, 2], [4, 5, 3], [7, 8, 6]].
    *   **Trạng thái đích (Goal State):** [[1, 2, 3], [4, 5, 6], [7, 8, 0]].
    *   **Các hành động/Phép toán (Actions/Operators):** Các di chuyển hợp lệ của ô trống (lên, xuống, trái, phải).
    *   **Mô hình chuyển đổi (Transition Model):** Mô tả trạng thái mới sẽ như thế nào sau khi thực hiện một hành động từ trạng thái hiện tại.
    *   **Chi phí đường đi (Path Cost):** Chi phí tích lũy của chuỗi hành động từ trạng thái bắt đầu đến trạng thái hiện tại (thường là 1 cho mỗi bước di chuyển ô trống trong 8-Puzzle).
    *   **Lời giải (Solution):** Một chuỗi các hành động từ trạng thái bắt đầu đến trạng thái đích. Một lời giải tối ưu là lời giải có tổng chi phí đường đi thấp nhất.

*   **Các thuật toán trong nhóm này được triển khai:**
    *   **Simple Hill Climbing:** Di chuyển đến hàng xóm đầu tiên tốt hơn trạng thái hiện tại.
    *   **Steepest Ascent Hill Climbing:** Di chuyển đến hàng xóm tốt nhất trong tất cả các hàng xóm của trạng thái hiện tại.
    *   **Stochastic Hill Climbing:** Chọn ngẫu nhiên một hàng xóm tốt hơn từ danh sách các hàng xóm tốt hơn.
    *   Gif dùng chung cho 3 thuật toán Hill Climbing
    ![](Gif/HC.gif)
    *   **Simulated Annealing (SA):** Tương tự Hill Climbing nhưng cho phép di chuyển đến trạng thái xấu hơn theo xác suất (giảm dần theo thời gian) để thoát khỏi cực tiểu cục bộ.
    *   **(số bước giải rất lớn, thời gian giải lâu nên phần này em không có gif)
    *   **Beam Search:** Duy trì một tập hợp (chùm) các trạng thái tốt nhất hiện tại (dựa trên heuristic) và mở rộng chúng ở mỗi bước, sau đó chỉ giữ lại những trạng thái tốt nhất từ các trạng thái mới sinh ra.
    ![](Gif/Beam.gif)

    *   **Genetic algorithm (GA):** Một phương pháp để giải quyết cả bài toán tối ưu hóa có ràng buộc và không ràng buộc dựa trên chọn lọc tự nhiên.
    ![](Gif/GA.gif)

*   **Hình ảnh so sánh hiệu suất:**
    *   **Simple Hill Climbing:
    ![](hieuSuat/SimpleHC.png)

    *   **Steepest Ascent Hill Climbing:
    ![](hieuSuat/SteppestHC.png)

    *   **Stochastic Hill Climbing:
    ![](hieuSuat/StochasticHC.png)

    *   **Simulated Annealing (SA):
    ![](hieuSuat/SA.png)

    *   **Beam Search:
    ![](hieuSuat/Beam.png)

    *   **Genetic Algorithm: 
    ![](hieuSuat/GA.png)

*   **Nhận xét về hiệu suất trong nhóm này khi áp dụng lên 8 ô chữ:**
    *   Về Độ dài đường đi (Path Steps):
        *   Simple HC, Steepest HC, Stochastic HC, và Beam Search tìm thấy đường đi có độ dài 4 bước.
        *   Genetic algorithm tìm thấy đường đi có độ dài 4 bước.
        *   Simulated Annealing tìm đường đi có độ dài 23962 bước.
    *   Về Số nút đã duyệt (Nodes):
        *   Simple HC, Steepest HC, và Stochastic HC duyệt ít nút nhất (4 nút).
        *   Beam Search duyệt số nút trung bình (12 nút).
        *   Genetic algorithm duyệt số nút khá nhiều (1128 nút).
        *   Simulated Annealing duyệt nhiều nút nhất (24510 nút).
    *   Về Thời gian thực thi (Time):
        *   Simple HC và Beam Search có thời gian thực thi nhanh nhất (0.000s).
        *   Steepest HC và Stochastic HC cũng rất nhanh (0.001s).
        *   Genetic algorithm tương đối nhanh (0.074s).
        *   Simulated Annealing có thời gian thực thi chậm nhất trong nhóm này (0.286s).
    *   Nhận xét tổng hợp: 
        *   Simple HC, Steepest HC, Stochastic HC, Genetic algorithm và Beam Search tìm thấy đường đi ngắn nhất (4 bước). Simulated Annealing tìm đường đi dài hơn đáng kể.
        *   Các thuật toán Hill Climbing (Simple, Steepest, Stochastic), Genetic algorithm và Beam Search thể hiện hiệu suất cao về thời gian và số nút duyệt trong trường hợp này.
        *   Simulated Annealing khám phá một không gian lớn hơn và có thời gian thực thi cao hơn.

### 2.4. Các thuật toán Tìm kiếm trong môi trường phức tạp

Nhóm này bao gồm các thuật toán tìm kiếm hoạt động trong các trạng thái mà agent không có đầy đủ thông tin.

#### 2.4.1. Tìm kiếm trên môi trường niềm tin (Belief Space Search / Sensorless Search)

Đây là nhóm thuật toán giải quyết các bài toán khi agent không biết chính xác trạng thái của mình, mà chỉ duy trì một tập hợp các trạng thái có thể (tập hợp niềm tin). Mục tiêu là tìm một chuỗi hành động đảm bảo đạt được trạng thái đích bất kể trạng thái ban đầu là gì (miễn là nó thuộc vào tập hợp niềm tin ban đầu). Các hành động được áp dụng cho toàn bộ tập hợp niềm tin.

*   **Các thành phần chính:**
    *   **Trạng thái niềm tin (Belief State):** Một tập hợp các trạng thái có thể của bàn cờ 8 ô chữ.
    *   **Trạng thái niềm tin ban đầu:** [[1, 2, 3], [4, 0, 5], [6, 7, 8]].
    *   **Hành động:** Một hành động được coi là khả thi trên không gian niềm tin nếu nó khả thi ở **tất cả** các trạng thái  trong tập hợp niềm tin hiện tại.
    *   **Mô hình chuyển đổi niềm tin:** Từ tập hợp niềm tin hiện tại, xác định tập hợp niềm tin mới bao gồm tất cả các trạng thái có thể đạt được.
    *   **Lời giải:** Một chuỗi hành động sao cho khi thực hiện chuỗi hành động này, tập hợp niềm tin cuối cùng chỉ chứa trạng thái đích.

*   **Các thuật toán trong nhóm này được triển khai:**

    *   **a. No Observation (Không quan sát được - triển khai bằng BFS):**
        *   **Mô tả:** Không nhận được bất kỳ thông tin phản hồi hay quan sát nào trong suốt quá trình thực hiện kế hoạch. Thuật toán phải tìm một kế hoạch đảm bảo đạt đích mà không cần biết mình đang ở trạng thái cụ thể nào sau mỗi bước đi. Triển khai sử dụng BFS trên không gian niềm tin để tìm kế hoạch ngắn nhất. Hàm `bfs_belief_search` thực hiện logic này.
        ![](Gif/Sensorless.gif)

    *   **b. Partially Observation (Quan sát được một phần - triển khai bằng BFS):**
        *   **Mô tả ý tưởng:** Nhận được quan sát sau mỗi bước đi. Quan sát này có thể giúp lọc và thu hẹp tập hợp niềm tin, giảm bớt sự không chắc chắn.
        ![](Gif/POsearch.gif)

*   **Hình ảnh so sánh hiệu suất:**
    *   No Observation (Không quan sát được):
    ![](hieuSuat/Sensorless.png)

    *   Partially Observation (Quan sát được một phần):
    ![](hieuSuat/POsearch.png)


*   Về Độ dài đường đi (Path Steps):
        *   No Observation và Partially Observation đều thực hiện 14 bước.
    *   Về Số các trạng thái đã tìm được:
        *   No Observation và Partially Observation đều thực hiện duyệt qua 6413 trạng thái.
    *   Về Thời gian tìm được trạng thái niềm tin phù hợp (Time):
        *   No Observation có thời gian tìm kiếm là 13.775s.
        *   Partially Observation có thời gian tìm kiếm nhanh hơn, là 6.117s.
    *   Nhận xét tổng hợp:
        *   Kích thước không gian niềm tin rất lớn, tìm kiếm trạng thái rất mất thời gian.
        *   Partially Observation có thêm sự hỗ trợ của các hàm lọc trạng thái và qua sát nên tốc độ tìm kiếm nhanh hơn No Observation.

    

#### 2.4.2. AND/OR (AOSerach)

Đây là một thuật toán tìm kiếm thử nghiệm được triển khai dựa trên logic AND/OR như đã thảo luận, áp dụng trực tiếp lên **không gian trạng thái vật lý**.

*   **Mô tả logic:**
    *   Thuật toán duyệt qua không gian trạng thái bằng cách xen kẽ giữa các nút "OR" và nút "AND" theo độ sâu.
    *   Nút OR: Thành công nếu *một trong* các nhánh con (hàng xóm) dẫn đến lời giải.
    *   Nút AND: Thành công nếu *tất cả* các nhánh con (hàng xóm) dẫn đến lời giải.
    *   Sử dụng cơ chế ngăn chặn chu trình trong đường đi khám phá hiện tại.

*   **Các thành phần chính của bài toán tìm kiếm:**
    *   **Trạng thái (State):** Một cấu hình cụ thể của bàn cờ 8 ô chữ.
    *   **Trạng thái bắt đầu (Initial State):** [[1, 2, 3], [4, 5, 6], [7, 0, 8]].
    *   **Trạng thái đích (Goal State):** [[1, 2, 3], [4, 5, 6], [7, 8, 0]].
    *   **Các hành động/Phép toán (Actions/Operators):** Các di chuyển hợp lệ của ô trống (lên, xuống, trái, phải).
    *   **Mô hình chuyển đổi (Transition Model): Mô tả cách một hành động cụ thể làm thay đổi trạng thái hiện tại để tạo ra trạng thái mới (được thực hiện bởi hàm apply_action và get_successors).
    *   **Lời giải (Solution):** Một chuỗi các hành động từ trạng thái bắt đầu đến trạng thái đích.

*   **Hình ảnh gif của thuật toán:**
    ![](Gif/AOsearch.gif)

*   **Hình ảnh hiệu suất:**
    ![](hieuSuat/AndOrSearch.png)

*   **Nhận xét về hiệu suất khi áp dụng lên 8 ô chữ:**
    *   Do bản chất của logic AND và cách xây dựng đường đi của nó không phù hợp với cấu trúc bài toán 8-Puzzle, thuật toán này sẽ **không tìm thấy lời giải hợp lý và tối ưu** cho các bài toán 8-Puzzle thông thường, chỉ tìm được lời giải cho trạng thái đơn giản như [[1, 2, 3], [4, 5, 6], [7, 0, 8]] .

### 2.5. Các Thuật toán tìm kiếm có ràng buộc (Constraint Satisfaction Problems - CSP)

Trong bài toán CSP, chúng ta cần tìm và gán giá trị cho một tập hợp các biến sao cho tất cả các ràng buộc giữa các biến được thỏa mãn. Mặc dù bài toán 8 ô chữ tìm đường đi không phải là CSP điển hình, các khái niệm và thuật toán CSP có thể được áp dụng cho các vấn đề liên quan như:

*   **Tạo một cấu hình bàn cờ 8 ô chữ hợp lệ:** Các biến là các ô trên bàn cờ, miền giá trị của mỗi biến là {0, 1, ..., 8}. Ràng buộc là mỗi giá trị từ 0 đến 8 phải xuất hiện đúng một lần trên bàn cờ.
*   **Tìm một trạng thái bắt đầu có thể giải được:** Sau khi tạo cấu hình hợp lệ, thêm ràng buộc rằng cấu hình đó phải có cùng tính chẵn lẻ nghịch thế với trạng thái đích (điều kiện để giải được).

*   **Các thành phần chính của bài toán CSP:**
    *   **Biến (Variables):** Tập hợp các yếu tố cần gán giá trị (9 ô trên bàn cờ 8-Puzzle).
    *   **Miền giá trị (Domains):** Tập hợp các giá trị có thể gán cho mỗi biến (ví dụ: {0, 1, 2, 3, 4, 5, 6, 7, 8} cho mỗi ô).
    *   **Ràng buộc (Constraints):** Các điều kiện mà các giá trị được gán phải thỏa mãn (ví dụ: mỗi số từ 0 đến 8 phải xuất hiện duy nhất một lần).

*   **Các thuật toán trong nhóm này được triển khai/áp dụng:**

    *   **Backtracking: Một thuật toán quay lui được sử dụng để sinh ra các hoán vị hợp lệ của bàn cờ (gán giá trị duy nhất từ 0-8 cho 9 ô). Quá trình này thử điền giá trị vào từng ô và quay lui nếu việc gán không thể hoàn thành một hoán vị hợp lệ.
    
    *   **Backtracking cho BFS (Thuật toán nhóm Uninformed Search):
    ![](Gif/BackTrackingBFS.gif)
    *   **Backtracking cho A* (Thuật toán nhóm Informed Search):
    ![](Gif/BackTrackingA.gif)
    *   **Backtracking cho Beam Search (Thuật toán nhóm Local Search):
    ![](Gif/BackTrackingBeam.gif)

    *   **AC3 (Arc Consistency 3): AC3 là một thuật toán duy trì tính nhất quán cung (arc consistency). Nó loại bỏ các giá trị khỏi miền giá trị của các biến nếu giá trị đó không thể thỏa mãn ràng buộc với bất kỳ giá trị nào trong miền giá trị của biến khác liên quan. AC3 được áp dụng cho ràng buộc "tất cả các giá trị phải khác nhau" giữa các ô.
    
    *   **AC3 cho BFS (Thuật toán nhóm Uninformed Search):
    ![](Gif/AC3BFS.gif)
    *   **AC3 cho A* (Thuật toán nhóm Informed Search):
    ![](Gif/AC3A.gif)
    *   **AC3 cho Beam Search (Thuật toán nhóm Local Search):
    ![](Gif/AC3Beam.gif)

*   **Hình ảnh so sánh hiệu suất:**
    *   **Backtracking cho BFS (Thuật toán nhóm Uninformed Search):
    ![](hieuSuat/BackTrackingBFS.png)

    *   **AC3 cho BFS (Thuật toán nhóm Uninformed Search):
    ![](hieuSuat/AC3BFS.png)

    *   **Backtracking cho A* (Thuật toán nhóm Informed Search):
    ![](hieuSuat/BackTrackingA.png)

    *   **AC3 cho A* (Thuật toán nhóm Informed Search):
    ![](hieuSuat/AC3A.png)

    *   **Backtracking cho Beam Search (Thuật toán nhóm Local Search):
    ![](hieuSuat/BackTrackingBeam.png)

    *   **AC3 cho Beam Search (Thuật toán nhóm Local Search):
    ![](hieuSuat/AC3Beam.png)

Dựa trên các kết quả bạn cung cấp cho quá trình sinh trạng thái bắt đầu, chúng ta có thể so sánh hiệu suất của phương pháp "Backtracking" và "AC3" thông qua ba thuật toán cụ thể là A*, BeamSearch, và BFS.

**So sánh Hiệu suất Sinh trạng thái: Backtracking vs AC3**

| Phương pháp sinh          | Thuật toán kiểm tra | Path Steps (Sinh) | Time (s) | Nodes (Sinh/Kiểm tra) |
|---------------------------|---------------------|-------------------|----------|-----------------------|
| **Backtracking**          | A\*                 | 50                | **0.004**| 242                   |
| AC3                       | A\*                 | 106               | 0.008    | 521                   |
|---------------------------|---------------------|-------------------|----------|-----------------------|
| **Backtracking**          | BeamSearch          | 22                | 1.133    | 81207                 |
| AC3                       | BeamSearch          | 25                | **2.188**| 154894                |
|---------------------------|---------------------|-------------------|----------|-----------------------|
| **Backtracking**          | BFS                 | 17                | **0.002**| 140                   |
| AC3                       | BFS                 | 22                | 0.055    | 1349                  |

**Nhận xét:**
1.  **Về Thời gian và Số nút:**
    *   Trong cả ba cặp so sánh (với A*, BeamSearch, BFS), phương pháp **"Backtracking" nhanh hơn** và **duyệt số nút ít hơn** so với phương pháp "AC3" tương ứng.
    *   Sự khác biệt rõ rệt nhất nằm ở các trường hợp kiểm tra bằng BeamSearch và BFS, nơi phương pháp Backtracking nhanh hơn đáng kể (1.133s vs 2.188s cho BeamSearch; 0.002s vs 0.055s cho BFS) và duyệt ít nút hơn nhiều (81207 vs 154894 cho BeamSearch; 140 vs 1349 cho BFS).
    *   Với A*, phương pháp Backtracking cũng nhanh hơn và duyệt ít nút hơn, dù sự khác biệt về thời gian không quá lớn.

2.  **Về Số bước sinh (Path Steps):**
    *   Trong hai trường hợp (với BeamSearch và BFS), phương pháp Backtracking tìm thấy trạng thái phù hợp với số bước sinh ít hơn (22 vs 25; 17 vs 22).
    *   Chỉ với A*, phương pháp Backtracking yêu cầu số bước sinh nhiều hơn đáng kể (50 vs 106). 

**Kết luận**: Dựa trên các kết quả cụ thể này, phương pháp "Backtracking" cho thấy hiệu suất tốt hơn trong việc sinh ra trạng thái bắt đầu phù hợp so với phương pháp "AC3", thể hiện qua thời gian thực thi nhanh hơn và số nút duyệt ít hơn trong hầu hết các trường hợp.

Tuyệt vời, bạn muốn thêm một mục mới về **Thuật toán tìm kiếm Học tăng cường (Reinforcement Learning Algorithms)** và trình bày thuật toán **Q-Learning** mà bạn đã triển khai.

Dựa trên code Q-Learning bạn vừa cung cấp, tôi sẽ soạn mục 2.6 này, trình bày khái niệm cơ bản của Học tăng cường và chi tiết về triển khai Q-Learning của bạn.

---

### 2.6. Các thuật toán Tìm kiếm Học tăng cường (Reinforcement Learning Algorithms)

Học tăng cường, nơi một agent học cách hành động trong một môi trường để tối đa hóa tổng phần thưởng nhận được theo thời gian. Agent học thông qua tương tác (thử và sai) với môi trường, thực hiện các hành động và nhận phản hồi dưới dạng phần thưởng hoặc hình phạt.

*   **Các thành phần chính:**
    *   **Môi trường (Environment):** Bài toán 8 ô chữ.
    *   **Agent:** Bộ giải 8-Puzzle, học cách chọn hành động.
    *   **Trạng thái (State):** Một cấu hình cụ thể của bàn cờ 8 ô chữ.
    *   **Hành động (Action):** Một di chuyển hợp lệ của ô trống.
    *   **Chính sách (Policy):** Chiến lược mà agent sử dụng để chọn hành động trong một trạng thái cụ thể.
    *   **Phần thưởng (Reward):** Phản hồi nhận được từ môi trường sau khi thực hiện một hành động (ví dụ: phần thưởng lớn khi đạt đích, hình phạt khi di chuyển không hợp lệ hoặc di chuyển xa đích).
    *   **Hàm giá trị (Value Function):** Dự đoán tổng phần thưởng tương lai mà agent có thể nhận được từ một trạng thái hoặc một cặp (trạng thái, hành động).

*   **Thuật toán trong nhóm này được triển khai:**

    *   **Q-Learning:**
        *   **Mô tả:** Q-Learning là một thuật toán học tăng cường không cần mô hình (model-free). Nó học một hàm giá trị hành động, gọi là Q-function (Q(s, a)), lưu trữ trong một bảng (Q-table). Q(s, a) ước tính tổng phần thưởng tương lai kỳ vọng khi thực hiện hành động `a` trong trạng thái `s` và sau đó đi theo chính sách tối ưu. Agent học Q-function thông qua thử và sai, cập nhật các giá trị trong Q-table dựa trên phần thưởng nhận được và giá trị Q ước tính của trạng thái tiếp theo.

        *   **Hình ảnh gif của thuật toán:**
        ![](Gif/QLearning.gif)

        *   **Hình ảnh so sánh hiệu suất:**
        ![](hieuSuat/QLearning.png)

        *   **Nhận xét về hiệu suất khi áp dụng lên 8 ô chữ:**
            *   Q-Learning là một thuật toán mạnh mẽ nhưng thường đòi hỏi một lượng lớn kinh nghiệm (nhiều episode và bước) để học được lời giải tốt cho các bài toán có không gian trạng thái lớn như 8-Puzzle, đặc biệt khi chỉ bắt đầu từ một trạng thái bắt đầu cố định.
            *   Hiệu quả phụ thuộc nhiều vào việc tinh chỉnh các tham số như `alpha`, `gamma`, `epsilon_decay`, `min_epsilon` và số lượng `episodes`.
            *   Đường đi trích xuất dựa trên Q-table đã học là đường đi theo chính sách tham lam tốt nhất mà agent tìm thấy, không nhất thiết là đường đi tối ưu toàn cục (như A*).
            *   Kích thước Q-table phản ánh số lượng các trạng thái-hành động mà agent đã ghé thăm và học được.

## 3. Kết luận

Qua quá trình xây dựng dự án, em đã đạt được một số kết quả. Em đã hiểu hơn về các thuật toán tìm kiếm, có cái nhìn mới mẻ về bộ môn trí tuệ nhân tạo.
Phát triển một ứng dụng GUI đầy đủ chức năng bằng Tkinter, cho phép người dùng tương tác để chọn thuật toán, xem trạng thái ban đầu/đích/hiện tại và quan sát quá trình giải bài toán 8 ô chữ một cách sinh động.
Trong dự án em đã phát triển một ứng dụng GUI đầy đủ chức năng bằng Tkinter, cho phép người dùng tương tác để chọn thuật toán, xem trạng thái ban đầu/đích/hiện tại và quan sát quá trình giải bài toán 8 ô chữ một cách sinh động. Triển khai tương đối thành công 6 nhóm thuật toán tìm kiếm, nhưng khả năng hoạt động vẫn chưa được tối ưu và còn thiếu sót.
* Tìm kiếm Không có thông tin: BFS, DFS, UCS, IDS.
* Tìm kiếm Có thông tin: A\*, Greedy Best-First, IDA*.
* Tìm kiếm Cục bộ: Simple Hill Climbing, Steepest Ascent Hill Climbing, Stochastic Hill Climbing, Simulated Annealing, Beam Search, Genetic Algorithm.
* Tìm kiếm Trong môi trường phức tạp: Search with no observation, Search with partialy observation, And Or Search.
* Tìm kiếm Có ràng buộc: Backtracking, AC3.
* Tìm kiếm Học tăng cường: Q-Learning.
Em đã trực tiếp quan sát hoạt động của các thuật toán, giúp hiểu hơn về lý thuyết và cách mà các thuật toán tìm kiếm.
Em đã triển khai hệ thống ghi lại các chỉ số hiệu suất (thời gian, nút duyệt, độ dài đường đi), qua đó thu thập được một cơ sở dữ liệu để dễ dàng có thể so sánh hiệu quả của các thuật toán.
Thông qua dự án này, em không chỉ nâng cao kiến thức về trí tuệ nhân tạo, mà còn rèn luyện được tư duy hệ thống, khả năng phân tích – đánh giá thuật toán, và kỹ năng xây dựng phần mềm có tính ứng dụng thực tiễn.
