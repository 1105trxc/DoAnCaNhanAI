Tuyệt vời, tôi sẽ cập nhật lại file `README.md` để phản ánh những thay đổi gần đây nhất trong code của bạn, đặc biệt là chức năng cài đặt trạng thái/niềm tin và danh sách thuật toán chính xác.

---

# 8-PUZZLE SOLVER VISUALIZATION (Tkinter)

Ứng dụng này là một công cụ tương tác được xây dựng bằng Python và thư viện Tkinter để giải bài toán 8-Puzzle bằng nhiều thuật toán tìm kiếm khác nhau và trực quan hóa quá trình giải đó. Nó cho phép người dùng cài đặt trạng thái bắt đầu, trạng thái kết thúc, và đặc biệt là tập hợp niềm tin ban đầu cho các thuật thuật tìm kiếm không cảm biến (sensorless search), cung cấp một cái nhìn sâu sắc hơn về các phương pháp giải quyết vấn đề trong AI.

## Giới thiệu về Bài toán 8-Puzzle

Bài toán 8-Puzzle là một trò chơi trượt ô cổ điển, thường được trình bày dưới dạng một khung hình vuông 3x3 với 8 ô vuông được đánh số (từ 1 đến 8) và một ô trống. Mục tiêu của trò chơi là sắp xếp lại các ô theo thứ tự tăng dần (hoặc một cấu hình đích khác) bằng cách trượt ô trống vào các vị trí lân cận (lên, xuống, trái, phải).

Đây là một bài toán điển hình trong lĩnh vực Trí tuệ Nhân tạo để minh họa các thuật toán tìm kiếm trong không gian trạng thái và không gian niềm tin.

**Trạng thái:** Một cấu hình của bảng 3x3.
**Hành động:** Di chuyển ô trống (lên, xuống, trái, phải).
**Trạng thái bắt đầu (State-Space):** Một cấu hình bảng ban đầu cho trước.
**Trạng thái đích (State-Space):** Một cấu hình bảng mong muốn.
**Niềm tin ban đầu (Belief-Space):** Một *tập hợp* các cấu hình bảng mà tác nhân có thể đang ở đó.
**Niềm tin đích (Belief-Space):** Một *tập hợp* chỉ chứa duy nhất trạng thái đích.

## Các Thuật toán Tìm kiếm đã triển khai

Ứng dụng này triển khai nhiều loại thuật toán tìm kiếm khác nhau:

### I. Thuật toán Tìm kiếm trên Không gian Trạng thái (State-Space Search)

Các thuật toán này giả định rằng tác nhân luôn biết chính xác mình đang ở trạng thái nào (observable).

1.  **BFS (Breadth-First Search):** Hoàn chỉnh, tối ưu (với chi phí bước đồng nhất). Khám phá theo chiều rộng.
2.  **DFS (Depth-First Search):** Hoàn chỉnh (với visited set và không gian hữu hạn), không tối ưu. Khám phá theo chiều sâu.
3.  **UCS (Uniform Cost Search):** Hoàn chỉnh, tối ưu. Mở rộng nút có chi phí thấp nhất.
4.  **Greedy Best-First Search:** Không hoàn chỉnh, không tối ưu. Mở rộng nút có heuristic tốt nhất (Manhattan).
5.  **A\* Search:** Hoàn chỉnh, tối ưu (với heuristic admissible và consistent). Mở rộng nút có f-cost (g+h) thấp nhất.
6.  **IDS (Iterative Deepening Search):** Hoàn chỉnh, tối ưu. Kết hợp ưu điểm bộ nhớ của DFS và tính hoàn chỉnh/tối ưu của BFS.
7.  **IDA\* (Iterative Deepening A\*):** Hoàn chỉnh, tối ưu. Kết hợp ưu điểm bộ nhớ của DFS và tính tối ưu của A\*. Thường hiệu quả cho 8-Puzzle.

### II. Thuật toán Tìm kiếm Cục bộ (Local Search)

Các thuật toán này không đảm bảo tìm được lời giải tối ưu hoặc tìm được lời giải nào cả, dễ bị kẹt ở điểm cực tiểu cục bộ.

1.  **Simple Hill Climbing:** Di chuyển đến trạng thái lân cận *đầu tiên* tốt hơn.
2.  **Steepest Ascent Hill Climbing:** Di chuyển đến trạng thái lân cận *tốt nhất* tốt hơn.
3.  **Random Hill Climbing:** Di chuyển đến trạng thái lân cận tốt hơn được *chọn ngẫu nhiên*.
4.  **Simulated Annealing (SA):** Có khả năng thoát local optima bằng cách chấp nhận ngẫu nhiên các bước đi tồi hơn với xác suất giảm dần.
5.  **Beam Search:** Giữ lại K trạng thái tốt nhất ở mỗi cấp độ. Không hoàn chỉnh, không tối ưu.

### III. Thuật toán Tìm kiếm trên Không gian Niềm tin (Belief-Space Search / Sensorless Search)

Các thuật toán này tìm kiếm một kế hoạch chung (chuỗi hành động cố định) hoạt động cho *tất cả* các trạng thái trong tập niềm tin ban đầu để đi đến niềm tin đích.

1.  **Sensorless Search (BFS on Belief Space):** Tìm kế hoạch ngắn nhất trong không gian niềm tin bằng BFS.
2.  **DFS\_Belief (DFS on Belief Space):** Tìm kế hoạch bằng DFS trong không gian niềm tin.

### IV. Thuật toán Khác

1.  **Backtracking Search (CSP Style):** Một hình thức tìm kiếm theo chiều sâu với quay lui để tìm đường đi trong không gian trạng thái.
2.  **Genetic Algorithm (GA):** Thuật toán tối ưu hóa dựa trên tiến hóa để tìm chuỗi hành động. (Lưu ý: Triển khai hiện tại tìm kế hoạch từ một trạng thái bắt đầu cụ thể, không phải tập niềm tin).

## Cách sử dụng

1.  Chạy file script Python.
2.  Giao diện chính hiển thị "Start", "End" và "Current" puzzle grids.
3.  Sử dụng các nút trong phần "Algorithms" để chạy các thuật toán tìm kiếm khác nhau.
4.  Sử dụng thanh trượt "Speed" để điều chỉnh tốc độ animation. Nút "Stop" để dừng animation.
5.  Sử dụng các nút trong phần "Setup":
    *   **Set Start:** Mở cửa sổ để nhập trạng thái bắt đầu tùy chỉnh.
    *   **Set Goal:** Mở cửa sổ để nhập trạng thái đích tùy chỉnh.
    *   **Set Belief:** Mở cửa sổ để quản lý (thêm, xóa, chỉnh sửa, đặt lại mặc định) tập hợp các trạng thái trong "Initial Belief Set" cho các thuật toán Sensorless/Belief Space.

## Cài đặt

Ứng dụng chỉ yêu cầu các module Python tích hợp sẵn (`tkinter`, `heapq`, `collections`, `random`, `threading`, `copy`, `functools`, `math`, `sys`).

Không cần cài đặt thêm thư viện bên ngoài.

## Yêu cầu hệ thống

*   Python 3.x
*   Hệ điều hành hỗ trợ Tkinter (Windows, macOS, Linux)

---