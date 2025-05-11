# 8-PUZZLE SOLVER VISUALIZATION (Tkinter)

Ứng dụng này là một công cụ tương tác được xây dựng bằng Python và thư viện Tkinter để giải bài toán 8-Puzzle bằng nhiều thuật toán tìm kiếm khác nhau và trực quan hóa quá trình giải đó. Nó cho phép người dùng cài đặt trạng thái bắt đầu, trạng thái kết thúc, và đặc biệt là tập hợp niềm tin ban đầu cho các thuật thuật tìm kiếm không cảm biến (sensorless search), cung cấp một cái nhìn sâu sắc hơn về các phương pháp giải quyết vấn đề trong AI.

**Trạng thái:** Một cấu hình của bảng 3x3.
**Hành động:** Di chuyển ô trống (lên, xuống, trái, phải).
**Trạng thái bắt đầu (State-Space):** Một cấu hình bảng ban đầu cho trước.
**Trạng thái đích (State-Space):** Một cấu hình bảng mong muốn.
**Niềm tin ban đầu (Belief-Space):** Một *tập hợp* các cấu hình bảng mà tác nhân có thể đang ở đó.
**Niềm tin đích (Belief-Space):** Một *tập hợp* chỉ chứa duy nhất trạng thái đích.

Dựa trên các thuật toán đã triển khai, chúng ta có thể phân chúng vào 5 nhóm:

1.  **Uninformed State Space Search (Tìm kiếm trên Không gian Trạng không có Thông tin):**
    *   Nhóm các thuật toán tìm kiếm trên không gian trạng thái mà **không sử dụng thông tin heuristic** (ước lượng khoảng cách đến mục tiêu) để hướng dẫn tìm kiếm.
    *   **Thuật toán từ code:**
        *   **DFS (Depth-First Search):** Duyệt sâu trước.
        *   **BFS (Breadth-First Search):** Duyệt rộng trước, tìm đường đi ngắn nhất (về số bước) khi chi phí bước đều nhau.
        *   **UCS (Uniform Cost Search):** Mở rộng nút có chi phí đường đi thực tế thấp nhất, tìm đường đi có chi phí thấp nhất.
        *   **DLS (Limited Depth Search):** Duyệt sâu có giới hạn độ sâu.
        *   **IDS (Iterative Deepening Depth-First Search):** Lặp lại DLS với độ sâu tăng dần.
        *   **Sensorless Search (BFS on Belief Space):** Tìm kiếm rộng trước trên không gian các tập hợp trạng thái có thể.
        *   **DFS on Belief Space:** Tìm kiếm sâu trước trên không gian các tập hợp trạng thái có thể.
2.  **Informed State Space Search (Tìm kiếm trên Không gian Trạng thái có Thông tin):**
    *   Nhóm các thuật thuật tìm kiếm trên không gian trạng thái mà **sử dụng thông tin heuristic** (ước lượng khoảng cách đến mục tiêu) để hướng dẫn tìm kiếm.
    *   **Thuật toán từ code:**
        *   **A\* (A\* Search):** Mở rộng nút dựa trên tổng chi phí thực tế đến nút đó cộng với chi phí ước lượng đến mục tiêu (g + h). Tìm đường đi tối ưu.
        *   **Greedy Best-First Search:** Mở rộng nút dựa hoàn toàn vào chi phí ước lượng đến mục tiêu (h). Không đảm bảo tối ưu.
        *   **IDA\* (Iterative Deepening A\* Search):** Lặp lại tìm kiếm giới hạn theo f-cost (g + h) tăng dần. Tìm đường đi tối ưu.

3.  **Local Search (Tìm kiếm Cục bộ):**
    *   Nhóm các thuật toán tìm kiếm lời giải bằng cách cải thiện ứng viên giải pháp hiện tại dựa trên các "lân cận" trong không gian tìm kiếm, không xây dựng cây tìm kiếm đầy đủ.
    *   **Thuật toán từ code:**
        *   **Simple Hill Climbing:** Từ điểm ngẫu nhiên, di chuyển đến lân cận tốt hơn đầu tiên.
        *   **Steepest Ascent Hill Climbing:** Từ điểm ngẫu nhiên, di chuyển đến lân cận tốt hơn có giá trị tốt nhất.
        *   **Random Hill Climbing:** Từ điểm ngẫu nhiên, di chuyển đến một lân cận tốt hơn được chọn ngẫu nhiên.
        *   **Simulated Annealing (SA):** Sử dụng nhiệt độ giảm dần để cho phép chấp nhận các nước đi xấu hơn với xác suất, giúp thoát khỏi cực trị cục bộ.
        *   **Beam Search:** Biến thể của Best-First Search, chỉ giữ lại một số lượng cố định (beam width) các trạng thái tốt nhất (theo heuristic) ở mỗi cấp độ. Không đảm bảo tối ưu hoặc hoàn chỉnh.
        *   **Genetic Algorithm (GA):** Tìm kiếm lời giải (chuỗi hành động) bằng cách thao tác trên một quần thể các chuỗi hành động thông qua các thế hệ.
        
4.  **CSPS (Constraint Satisfaction Problem Solving):**
    *   Nhóm các thuật toán giải các bài toán tìm kiếm cấu hình thỏa mãn một tập hợp các ràng buộc. Trong code này, CSPS được dùng để giải bài toán phụ: tìm một cấu hình ma trận 3x3 chứa các số từ 0 đến 8 duy nhất một lần (tức là một trạng thái 8-Puzzle hợp lệ).
    *   **Thuật toán từ code:**
        *   **Backtracking:** Phương pháp có hệ thống để xây dựng từng phần của giải pháp và quay lui khi vi phạm ràng buộc. (Hàm `backtracking` trong code sử dụng Backtracking Search để *tìm một trạng thái bắt đầu* hợp lệ).

5.  **Complex Environment (Tìm kiếm trong môi trường phức tạp):**
    *   Một loại tìm kiếm trong môi trường phức tạp, không xác, nhiều tác nhân, trạng thái.
    *   **Thuật toán từ code:**
        *   AND-OR Search (AOSearch): Tìm kiếm bằng cách xây dựng cây kế hoạch gồm các node AND và OR nhằm xử lý mọi khả năng có thể xảy ra.



