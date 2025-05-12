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
    ![](hieSuat/DFS.png)
    *   **UCS (Uniform Cost Search)
    ![](hieSuat/UCS.png)
    *   **IDS (Iterative Deepening Search)
    ![](hieSuat/IDS.png)

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
    ![](hieSuat/Greedy.png)
    *   IDA\* (Iterative Deepening A\*):
    ![](hieSuat/IDA_star.png)
   
*   **Nhận xét về hiệu suất trong nhóm này khi áp dụng lên 8 ô chữ:**
    *   Về Độ dài đường đi (Path Steps):
        *   A* và IDA* tìm thấy đường đi có độ dài 23 bước. Đây là độ dài tối ưu cho bài toán này.
        *   Greedy Best-First Search tìm đường đi có độ dài 79 bước.
    *   Về Số nút đã duyệt (Nodes):
        *   A* là nhanh thứ hai (0.013s).
        *   Greedy Best-First Search là nhanh nhất (0.007s).
        *   IDA* chậm hơn A* một chút (0.023s).
    *   Về Thời gian thực thi (Time):
        *   DFS là thuật toán nhanh nhất (0.604 giây).
        *   BFS (1.297 giây) nhanh hơn UCS (1.880 giây).
        *   IDS là thuật toán chậm nhất (6.926 giây).
    *   Nhận xét tổng hợp: BFS, UCS và IDS tìm thấy đường đi tối ưu (23 bước). DFS tìm đường đi không tối ưu (7113 bước). DFS là nhanh nhất và duyệt ít nút nhất trong trường hợp này, trong khi IDS là chậm nhất và duyệt nhiều nút nhất.
        *   A* và IDA* tìm thấy lời giải tối ưu. Greedy Best-First Search không đảm bảo tính tối ưu.
        *   Greedy Best-First Search hiệu quả nhất về thời gian và số nút duyệt trong trường hợp này.

### 2.3. Các thuật toán Tìm kiếm cục bộ (Local Search)

Các thuật toán này thường chỉ duy trì một hoặc một vài trạng thái hiện tại và di chuyển đến các trạng thái lân cận dựa trên một tiêu chí (thường là cải thiện giá trị mục tiêu/heuristic). Chúng không ghi nhớ đường đi đầy đủ từ điểm bắt đầu và có thể mắc kẹt ở cực tiểu cục bộ hoặc không tìm thấy lời giải.

*   **Các thuật toán trong nhóm này được triển khai:**
    *   **Simple Hill Climbing:** Di chuyển đến hàng xóm đầu tiên tốt hơn trạng thái hiện tại.
    *   **Steepest Ascent Hill Climbing:** Di chuyển đến hàng xóm tốt nhất trong tất cả các hàng xóm của trạng thái hiện tại.
    *   **Random Hill Climbing:** Chọn ngẫu nhiên một hàng xóm tốt hơn từ danh sách các hàng xóm tốt hơn.
    *   **Simulated Annealing (SA):** Tương tự Hill Climbing nhưng cho phép di chuyển đến trạng thái xấu hơn theo xác suất (giảm dần theo thời gian/nhiệt độ) để thoát khỏi cực tiểu cục bộ.
    *   **Beam Search:** Duy trì một tập hợp (chùm) các trạng thái tốt nhất hiện tại (dựa trên heuristic) và mở rộng chúng ở mỗi bước, sau đó chỉ giữ lại những trạng thái tốt nhất từ các trạng thái mới sinh ra.
    ![](Gif/Beam.gif)

*   **Hình ảnh gif của từng thuật toán:**
    *   *(Chèn GIF trực quan cho Simple Hill Climbing tại đây)*
    *   *(Chèn GIF trực quan cho Steepest Ascent Hill Climbing tại đây)*
    *   *(Chèn GIF trực quan cho Random Hill Climbing tại đây)*
    *   *(Chèn GIF trực quan cho Simulated Annealing tại đây)*
    *   *(Chèn GIF trực quan cho Beam Search tại đây)*

*   **Hình ảnh so sánh hiệu suất:**
    *   *(Thu thập dữ liệu. Báo cáo Thời gian, Số nút duyệt/bước nhảy, và trạng thái cuối cùng đạt được (giá trị heuristic cuối cùng). Ghi rõ nếu không tìm thấy lời giải.)*

*   **Nhận xét về hiệu suất trong nhóm này khi áp dụng lên 8 ô chữ:**
    *   Rất nhanh và hiệu quả về bộ nhớ, vì chúng không lưu trữ toàn bộ không gian tìm kiếm hoặc đường đi đầy đủ.
    *   **Không đảm bảo tính đầy đủ hoặc tối ưu.** Thường chỉ hiệu quả trong việc tìm kiếm lời giải "đủ tốt" hoặc nhanh chóng cho các bài toán có nhiều lời giải hoặc khi không yêu cầu tối ưu.
    *   Hill Climbing variants rất dễ mắc kẹt ở cực tiểu cục bộ.
    *   SA và Beam Search cố gắng cải thiện khả năng thoát khỏi cực tiểu cục bộ/khám phá rộng hơn nhưng vẫn không đảm bảo tìm thấy lời giải tối ưu hoặc tìm thấy lời giải cho mọi trường hợp có lời giải.

### 2.4. Các thuật toán Tìm kiếm không cảm biến (Sensorless Search)

Tìm kiếm trong Không gian Niềm tin (Belief Space). Agent không biết chính xác trạng thái vật lý của mình, mà chỉ biết nó nằm trong một tập hợp các trạng thái có thể (tập hợp niềm tin). Mục tiêu là tìm một chuỗi hành động đảm bảo đạt đích bất kể trạng thái vật lý ban đầu là gì (miễn là thuộc vào tập hợp niềm tin ban đầu).

*   **Các thành phần chính:**
    *   **Trạng thái niềm tin (Belief State):** Một tập hợp các trạng thái vật lý có thể của bàn cờ.
    *   **Trạng thái niềm tin ban đầu:** Tập hợp các trạng thái vật lý mà agent có thể bắt đầu.
    *   **Hành động:** Một hành động vật lý (di chuyển ô trống). Một hành động khả thi trên không gian niềm tin nếu nó khả thi ở **tất cả** các trạng thái vật lý trong tập hợp niềm tin hiện tại.
    *   **Mô hình chuyển đổi niềm tin:** Từ tập hợp niềm tin hiện tại và một hành động, xác định tập hợp niềm tin mới (tập hợp tất cả các trạng thái có thể đạt được).
    *   **Lời giải:** Một chuỗi hành động (kế hoạch) sao cho tập hợp niềm tin cuối cùng chỉ chứa (các) trạng thái đích.

*   **Các thuật toán trong nhóm này được triển khai:**
    *   **BFS on Belief Space:** Tìm kiếm theo chiều rộng trên không gian niềm tin.
    *   **DFS on Belief Space:** Tìm kiếm theo chiều sâu trên không gian niềm tin (có thể dùng giới hạn độ sâu).

*   **Hình ảnh gif của từng thuật toán:**
    *   *(Chèn GIF trực quan cho BFS on Belief Space tại đây. Hoạt họa sẽ hiển thị quá trình mô phỏng kế hoạch.)*
    *   *(Chèn GIF trực quan cho DFS on Belief Space tại đây. Hoạt họa sẽ hiển thị quá trình mô phỏng kế hoạch.)*

*   **Hình ảnh so sánh hiệu suất:**
    *   *(Thu thập dữ liệu. Báo cáo Thời gian, Số tập hợp niềm tin đã duyệt, và Độ dài kế hoạch (số hành động).)*

*   **Nhận xét về hiệu suất trong nhóm này khi áp dụng lên 8 ô chữ:**
    *   Khó khăn chính là kích thước không gian niềm tin có thể rất lớn.
    *   Kế hoạch tìm được có tính mạnh mẽ (robust) cao hơn so với đường đi trạng thái thông thường (đảm bảo đạt đích nếu trạng thái ban đầu thuộc tập niềm tin ban đầu).
    *   Thường tốn kém hơn tìm kiếm trạng thái vật lý nếu không gian niềm tin phát triển nhanh. BFS trên không gian niềm tin có thể tìm kế hoạch ngắn nhất.

### 2.5. Thuật toán Di truyền (Genetic Algorithm - GA)

Một thuật toán tìm kiếm tối ưu hóa dựa trên cơ chế tiến hóa tự nhiên. Nó duy trì một quần thể các "cá thể" (các giải pháp tiềm năng, ở đây là các chuỗi hành động), đánh giá "độ thích nghi" (fitness) của chúng dựa trên mức độ tốt của giải pháp (ví dụ: mức độ gần đích của trạng thái cuối cùng khi thực hiện chuỗi hành động), và tạo ra thế hệ mới bằng cách chọn lọc các cá thể tốt, lai ghép (kết hợp giải pháp) và đột biến (thay đổi ngẫu nhiên).

*   **Các thành phần chính:** Quần thể (Population), Cá thể (Individual - Action Sequence), Hàm thích nghi (Fitness Function), Chọn lọc (Selection), Lai ghép (Crossover), Đột biến (Mutation), Thế hệ (Generation).

*   **Thuật toán được triển khai:** Genetic Algorithm cơ bản áp dụng để tìm chuỗi hành động giải 8-Puzzle.

*   **Hình ảnh gif của thuật toán:**
    *   *(Chèn GIF trực quan cho Genetic Algorithm tại đây. Hoạt họa thường hiển thị đường đi mô phỏng của cá thể tốt nhất tìm được sau mỗi thế hệ hoặc cuối cùng.)*

*   **Hình ảnh so sánh hiệu suất:**
    *   *(Thu thập dữ liệu. Báo cáo Thời gian, Số lần đánh giá cá thể (Evaluations), Số thế hệ, và Độ thích nghi tốt nhất đạt được. GA không đảm bảo tìm thấy lời giải, báo cáo trạng thái cuối cùng và H của cá thể tốt nhất.)*

*   **Nhận xét về hiệu suất khi áp dụng lên 8 ô chữ:**
    *   Là một thuật toán ngẫu nhiên và heuristic, không đảm bảo tính đầy đủ hoặc tối ưu.
    *   Có thể hiệu quả cho các không gian tìm kiếm rất lớn hoặc phức tạp nơi các thuật toán tìm kiếm truyền thống gặp khó khăn, nhưng với 8-Puzzle, thường chậm hơn và kém đảm bảo hơn A\* hoặc IDA\*.
    *   Hiệu suất phụ thuộc nhiều vào việc tinh chỉnh các tham số (kích thước quần thể, tỷ lệ đột biến, cách chọn lọc, hàm thích nghi).

### 2.6. Chức năng Sinh Trạng thái Bắt đầu và Thuật toán Thử nghiệm

Phần này bao gồm các chức năng bổ trợ hoặc thuật toán thử nghiệm không thuộc 5 nhóm tìm kiếm tiêu chuẩn ở trên.

*   **Sinh Trạng thái Bắt đầu (Backtracking State Generation):**
    *   **Mô tả:** Chức năng này được kích hoạt bởi nút "Backtracking" trên GUI. Nó không phải là thuật toán giải bài toán 8-Puzzle, mà là một công cụ để tạo ra một trạng thái bắt đầu mới cho câu đố. Nó sử dụng kỹ thuật sinh ngẫu nhiên các hoán vị của các số từ 0 đến 8 và sau đó kiểm tra tính giải được của hoán vị đó so với trạng thái đích bằng quy tắc số nghịch thế.
    *   *(Mô tả các phương pháp sinh (Random Solvable State, Backward Random Walk) nếu bạn giữ các tùy chọn đó.)*
    *   *(Chèn hình ảnh/GIF minh họa việc tạo trạng thái mới trên GUI)*
    *   **Nhận xét:** Chức năng này đảm bảo rằng mọi trạng thái bắt đầu mới được thiết lập trên GUI đều là một bài toán có lời giải, cho phép người dùng thử nghiệm các thuật toán giải trên các trường hợp đảm bảo có lời giải.

*   **Thuật toán Thử nghiệm Custom AND/OR (AOSerach):**
    *   **Mô tả:** Đây là một thuật toán tìm kiếm được triển khai dựa trên logic AND/OR như bạn đã cung cấp, áp dụng lên không gian trạng thái của bài toán 8-Puzzle.
    *   *(Chèn GIF trực quan nếu nó tạo ra đường đi nào đó, ngay cả khi không đến đích.)*
    *   **Nhận xét:** Cần nhấn mạnh lại rằng thuật toán này không phải là cách áp dụng chuẩn mực của AND/OR Search cho bài toán 8-Puzzle tìm đường đi. Do bản chất của logic AND và cách xây dựng đường đi của nó không phù hợp với không gian trạng thái phẳng, nó dự kiến sẽ không tìm thấy lời giải hợp lý hoặc tối ưu cho các bài toán 8-Puzzle thông thường. Nó được giữ lại trong ứng dụng như một minh họa cho một cách tiếp cận tìm kiếm khác (mặc dù không hiệu quả cho bài toán này).

## 3. Kết luận

Dự án "8-Puzzle Solver Visualization" là một công cụ học tập hữu ích, cung cấp góc nhìn trực quan về hoạt động của nhiều thuật toán tìm kiếm AI kinh điển. Bằng cách trực quan hóa quá trình duyệt trạng thái và so sánh các chỉ số hiệu suất, người dùng có thể hiểu sâu sắc hơn về điểm mạnh, điểm yếu và phạm vi ứng dụng của từng thuật toán trong việc giải quyết bài toán 8 ô chữ. Từ các phương pháp vét cạn đơn giản đến các kỹ thuật heuristic thông minh và các phương pháp metaheuristic, ứng dụng minh họa rõ ràng tầm quan trọng của việc lựa chọn đúng thuật toán và cấu trúc dữ liệu phù hợp với bản chất của bài toán.
