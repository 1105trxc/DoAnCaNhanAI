# 8-PUZZLE SOLVER VISUALIZATION

Ứng dụng này là một công cụ tương tác được xây dựng bằng Python và thư viện Tkinter để giải bài toán 8-Puzzle bằng nhiều thuật toán tìm kiếm khác nhau và trực quan hóa quá trình giải đó. Nó cho phép người dùng cài đặt trạng thái bắt đầu, trạng thái kết thúc, và thậm chí cả tập hợp niềm tin ban đầu cho các thuật toán tìm kiếm không cảm biến (sensorless search).

## Giới thiệu về Bài toán 8-Puzzle

Bài toán 8-Puzzle là một trò chơi trượt ô cổ điển, thường được trình bày dưới dạng một khung hình vuông 3x3 với 8 ô vuông được đánh số (từ 1 đến 8) và một ô trống. Mục tiêu của trò chơi là sắp xếp lại các ô theo thứ tự tăng dần bằng cách trượt ô trống vào các vị trí lân cận (lên, xuống, trái, phải).

Đây là một bài toán điển hình trong lĩnh vực Trí tuệ Nhân tạo để minh họa các thuật toán tìm kiếm trong không gian trạng thái.

**Trạng thái:** Một cấu hình của bảng 3x3 (vị trí của 8 ô số và 1 ô trống).
**Hành động:** Di chuyển ô trống lên, xuống, trái, hoặc phải (nếu khả thi).
**Trạng thái bắt đầu:** Một cấu hình bảng ban đầu cho trước.
**Trạng thái đích:** Một cấu hình bảng mong muốn (thường là các số được sắp xếp theo thứ tự).

## Các Thuật toán Tìm kiếm đã triển khai

Ứng dụng này triển khai nhiều loại thuật toán tìm kiếm khác nhau, bao gồm tìm kiếm trên không gian trạng thái (state-space search) và tìm kiếm trên không gian niềm tin (belief-space search - sensorless search).

### I. Thuật toán Tìm kiếm trên Không gian Trạng thái (State-Space Search)

Các thuật toán này giả định rằng tác nhân luôn biết chính xác mình đang ở trạng thái nào (observable).

1.  **BFS (Breadth-First Search):**
    *   **Nguyên lý:** Khám phá đồ thị theo chiều rộng, thăm tất cả các nút ở độ sâu `d` trước khi chuyển sang độ sâu `d+1`.
    *   **Tính chất:** Hoàn chỉnh (complete) nếu có lời giải, tối ưu (optimal) nếu chi phí bước là đồng nhất (như trong 8-Puzzle).
    *   **Nhược điểm:** Có thể sử dụng lượng bộ nhớ rất lớn trên các bài toán có yếu tố phân nhánh cao (high branching factor).

2.  **DFS (Depth-First Search):**
    *   **Nguyên lý:** Khám phá đồ thị theo chiều sâu, đi sâu vào một nhánh càng xa càng tốt trước khi quay lui.
    *   **Tính chất:** Không hoàn chỉnh trên đồ thị có chu kỳ (không có visited set), không tối ưu. Hoàn chỉnh trên cây hữu hạn.
    *   **Nhược điểm:** Dễ bị kẹt trong các nhánh sâu vô tận hoặc không có lời giải.

3.  **UCS (Uniform Cost Search):**
    *   **Nguyên lý:** Mở rộng nút có chi phí đường đi từ gốc (g-cost) thấp nhất.
    *   **Tính chất:** Hoàn chỉnh, tối ưu.
    *   **Nhược điểm:** Giống BFS, có thể sử dụng bộ nhớ lớn. Không sử dụng thông tin heuristic để dẫn đường.

4.  **Greedy Best-First Search:**
    *   **Nguyên lý:** Mở rộng nút có giá trị heuristic (h-cost) thấp nhất. Sử dụng heuristic Manhattan để đánh giá khoảng cách đến đích.
    *   **Tính chất:** Không hoàn chỉnh, không tối ưu.
    *   **Ưu điểm:** Có thể tìm thấy lời giải nhanh hơn BFS/UCS nếu heuristic tốt, sử dụng ít bộ nhớ hơn (nếu không lưu visited set).

5.  **A\* Search:**
    *   **Nguyên lý:** Mở rộng nút có giá trị f-cost (f = g + h) thấp nhất, trong đó g là chi phí đường đi từ gốc và h là heuristic đến đích. Sử dụng heuristic Manhattan.
    *   **Tính chất:** Hoàn chỉnh, tối ưu nếu heuristic là admissible (heuristic Manhattan là admissible).
    *   **Ưu điểm:** Kết hợp ưu điểm của UCS (tối ưu) và Greedy (sử dụng heuristic để dẫn đường). Thường là một trong những thuật toán hiệu quả nhất cho bài toán 8-Puzzle.

6.  **IDS (Iterative Deepening Search):**
    *   **Nguyên lý:** Thực hiện một loạt các tìm kiếm theo chiều sâu với giới hạn độ sâu tăng dần (Depth-Limited Search).
    *   **Tính chất:** Hoàn chỉnh, tối ưu (nếu chi phí bước là đồng nhất). Có hiệu quả về bộ nhớ như DFS.
    *   **Ưu điểm:** Kết hợp ưu điểm về bộ nhớ của DFS và tính hoàn chỉnh/tối ưu của BFS (trên đồ thị không trọng số).

7.  **IDA\* (Iterative Deepening A\*):**
    *   **Nguyên lý:** Thực hiện một loạt các tìm kiếm theo chiều sâu với giới hạn chi phí (threshold) tăng dần. Giới hạn chi phí ban đầu bằng heuristic của trạng thái gốc, sau đó tăng lên ngưỡng f-cost nhỏ nhất vượt quá ngưỡng hiện tại trong lần lặp trước. Sử dụng heuristic Manhattan.
    *   **Tính chất:** Hoàn chỉnh, tối ưu nếu heuristic là admissible và consistent. Hiệu quả về bộ nhớ như DFS.
    *   **Ưu điểm:** Kết hợp ưu điểm về bộ nhớ của DFS và tính tối ưu của A\*. Thường là một trong những thuật toán hiệu quả nhất về cả thời gian và bộ nhớ cho bài toán 8-Puzzle.

### II. Thuật toán Tìm kiếm Cục bộ (Local Search)

Các thuật toán này không khám phá toàn bộ không gian trạng thái mà tập trung vào việc cải thiện dần trạng thái hiện tại dựa trên thông tin lân cận. Chúng không đảm bảo tìm được lời giải tối ưu hoặc tìm được lời giải.

1.  **Simple Hill Climbing:**
    *   **Nguyên lý:** Di chuyển đến trạng thái lân cận *đầu tiên* có heuristic tốt hơn trạng thái hiện tại.
    *   **Tính chất:** Không hoàn chỉnh, không tối ưu.
    *   **Nhược điểm:** Dễ bị kẹt ở local optima hoặc plateau.

2.  **Steepest Ascent Hill Climbing:**
    *   **Nguyên lý:** Di chuyển đến trạng thái lân cận *tốt nhất* (có heuristic thấp nhất) trong số các trạng thái lân cận có heuristic tốt hơn trạng thái hiện tại.
    *   **Tính chất:** Không hoàn chỉnh, không tối ưu.
    *   **Nhược điểm:** Cũng dễ bị kẹt ở local optima hoặc plateau.

3.  **Random Hill Climbing:**
    *   **Nguyên lý:** Chọn ngẫu nhiên một trạng thái lân cận *trong số các trạng thái tốt hơn* trạng thái hiện tại.
    *   **Tính chất:** Không hoàn chỉnh, không tối ưu.
    *   **Nhược điểm:** Có thể thoát khỏi các plateau tốt hơn Simple/Steepest một chút do tính ngẫu nhiên, nhưng vẫn dễ bị kẹt ở local optima.

4.  **Simulated Annealing (SA):**
    *   **Nguyên lý:** Dựa trên quá trình luyện kim. Di chuyển đến trạng thái tốt hơn luôn được chấp nhận. Di chuyển đến trạng thái tồi hơn có thể được chấp nhận với xác suất giảm dần theo "nhiệt độ" (giảm dần theo thời gian).
    *   **Tính chất:** Có thể thoát khỏi local optima. Hoàn chỉnh (với lịch trình làm nguội phù hợp và thời gian vô hạn), nhưng không tối ưu.
    *   **Ưu điểm:** Có khả năng tìm được lời giải tốt hơn Hill Climbing bằng cách chấp nhận rủi ro di chuyển đến trạng thái tồi hơn.

5.  **Beam Search:**
    *   **Nguyên lý:** Mở rộng K trạng thái tốt nhất ở mỗi cấp độ tìm kiếm. Giữ K trạng thái tốt nhất trong số các trạng thái con được sinh ra làm "chùm tia" cho cấp độ tiếp theo.
    *   **Tính chất:** Không hoàn chỉnh, không tối ưu.
    *   **Ưu điểm:** Kiểm soát bộ nhớ (chỉ lưu K trạng thái mỗi cấp). Có thể tìm lời giải nhanh hơn BFS/UCS nếu K đủ lớn và heuristic tốt.

### III. Thuật toán Tìm kiếm trên Không gian Niềm tin (Belief-Space Search / Sensorless Search)

Các thuật toán này hoạt động khi tác nhân không biết chính xác trạng thái ban đầu của mình (chỉ biết nó nằm trong một tập hợp các trạng thái - trạng thái niềm tin). Mục tiêu là tìm một kế hoạch chung (chuỗi hành động cố định) mà khi áp dụng cho *tất cả* các trạng thái trong tập niềm tin ban đầu, đều đưa tác nhân đến **duy nhất trạng thái đích** (để tác nhân biết chắc mình đã đến đích).

1.  **Sensorless Search (BFS on Belief Space):**
    *   **Nguyên lý:** Thực hiện tìm kiếm BFS trên không gian của các *tập hợp trạng thái* (trạng thái niềm tin). Một hành động chỉ khả thi cho một trạng thái niềm tin nếu nó hợp lệ cho *tất cả* các trạng thái riêng lẻ trong tập hợp niềm tin đó.
    *   **Tính chất:** Hoàn chỉnh (nếu không gian niềm tin là hữu hạn và có kế hoạch giải), tối ưu (tìm kế hoạch ngắn nhất) trong không gian niềm tin.
    *   **Nhược điểm:** Không gian niềm tin là rất lớn. Tìm kiếm có thể rất chậm hoặc hết bộ nhớ/thời gian. Việc tìm một kế hoạch chung hoạt động cho nhiều trạng thái ban đầu là rất khó.

2.  **DFS\_Belief (DFS on Belief Space):**
    *   **Nguyên lý:** Thực hiện tìm kiếm DFS trên không gian niềm tin.
    *   **Tính chất:** Hoàn chỉnh (trên đồ thị niềm tin hữu hạn nếu không có chu kỳ hoặc có visited set), không tối ưu (tìm kế hoạch đầu tiên tìm thấy).
    *   **Nhược điểm:** Giống như BFS Belief, không gian niềm tin lớn. Có thể tìm kế hoạch nhanh hơn BFS Belief nếu kế hoạch đầu tiên nằm ở nhánh sâu được thăm trước.

3.  **Backtracking Search (CSP Style):**
    *   **Nguyên lý:** Sử dụng một hình thức tìm kiếm theo chiều sâu với quay lui để tìm một đường đi từ trạng thái bắt đầu đến trạng thái đích trong không gian trạng thái. Phiên bản này được trình bày theo phong cách giải bài toán CSP (Constraint Satisfaction Problem), nơi mỗi bước di chuyển là một "assignment" và việc kiểm tra chu kỳ là một dạng "constraint".
    *   **Tính chất:** Hoàn chỉnh (nếu có giới hạn độ sâu phù hợp hoặc không có chu kỳ), không tối ưu (tìm đường đi đầu tiên).
    *   **Nhược điểm:** Dễ bị kẹt trong đệ quy sâu hoặc khám phá không gian lớn nếu không có heuristic hoặc cắt tỉa hiệu quả.

### IV. Thuật toán Di truyền (Genetic Algorithm - GA)

GA là một thuật toán tối ưu hóa dựa trên quá trình tiến hóa tự nhiên. Trong ứng dụng này, nó được sử dụng để tìm một chuỗi hành động giải puzzle.

1.  **Genetic Algorithm (GA):**
    *   **Nguyên lý:** Duy trì một quần thể các "cá thể" (mỗi cá thể là một chuỗi hành động). Các cá thể được đánh giá bằng hàm "fitness" (độ tốt của trạng thái cuối cùng sau khi thực hiện chuỗi hành động). Quần thể tiến hóa qua các thế hệ bằng cách chọn lọc các cá thể tốt hơn, lai ghép (crossover) các chuỗi hành động, và đột biến (mutation) ngẫu nhiên các hành động.
    *   **Tính chất:** Là thuật toán tìm kiếm xác suất (probabilistic search). Không hoàn chỉnh, không tối ưu.
    *   **Ưu điểm:** Có thể tìm được lời giải trong không gian tìm kiếm rất lớn mà các thuật toán tìm kiếm đồ thị truyền thống không hiệu quả.
    *   **Nhược điểm:** Không đảm bảo tìm được lời giải, có thể dừng lại ở một giải pháp không tối ưu hoặc không tìm được giải pháp nào cả. Cần nhiều tham số điều chỉnh (kích thước quần thể, tỉ lệ đột biến, v.v.). *Lưu ý: Trong code này, GA được triển khai để tìm kế hoạch từ một trạng thái bắt đầu *cụ thể* (không phải từ tập niềm tin).*


## Cài đặt

Ứng dụng chỉ yêu cầu thư viện Tkinter (thường có sẵn trong cài đặt Python chuẩn) và các module Python tích hợp sẵn (`heapq`, `collections`, `random`, `threading`, `copy`, `math`, `sys`).

Không cần cài đặt thêm thư viện bên ngoài.

