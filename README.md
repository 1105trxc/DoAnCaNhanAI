Dưới đây là nội dung của tệp tin được chuyển đổi sang định dạng Markdown theo yêu cầu của bạn:

[cite_start]**Nhóm thực hiện: 01** [cite: 961]
[cite_start]**Thành viên nhóm:** [cite: 962]

| Họ tên | Mã số sinh viên | Nhiệm vụ (chức năng) |
| :--- | :--- | :--- |
| Nguyễn Thành Tin | 23110340 | [cite_start]Đăng ký [cite: 963] |
| Trác Ngọc Đăng Khoa | 23110243 | [cite_start]Đăng nhập [cite: 963] |
| Phan Đình Duẩn | 23110192 | [cite_start]Quên mật khẩu [cite: 963] |
| Nguyễn Duy Cường | 23110189 | [cite_start]Cập nhật thông tin [cite: 963] |

---

### [cite_start]1. Chức năng Đăng ký tài khoản [cite: 964]
#### 1.1. [cite_start]Đặc tả UC01: Đăng ký tài khoản (Register) [cite: 965]

| Trường | Nội dung |
| :--- | :--- |
| **Use Case ID** | [cite_start]UC01 [cite: 966] |
| **Use Case Name** | [cite_start]Đăng ký tài khoản [cite: 966] |
| **Description** | Là một Khách vãng lai, tôi muốn đăng ký tài khoản thành viên thông qua mã xác thực OTP gửi về email. [cite_start]Hệ thống có cơ chế bảo vệ biểu mẫu khỏi việc đăng ký tự động/spam. [cite: 966] |
| **Actor(s)** | [cite_start]Khách vãng lai (Guest) [cite: 966] |
| **Priority** | [cite_start]Must Have [cite: 966] |
| **Trigger** | [cite_start]Người dùng nhấn nút "Đăng ký" trên giao diện. [cite: 966] |
| **Pre-Condition(s)** | [cite_start]Người dùng chưa đăng nhập vào hệ thống. [cite: 966] |
| **Post-Condition(s)** | [cite_start]Tài khoản được tạo thành công, người dùng được chuyển trạng thái thành Khách hàng (User) và được cấp phiên làm việc. [cite: 966] |
| **Basic Flow** | 1. Người chọn lệnh "Đăng ký" trên màn hình.<br>2. Hệ thống hiển thị biểu mẫu yêu cầu cung cấp thông tin (Họ tên, Email, Mật khẩu).<br>3. Người dùng nhập thông tin và nhấn "Gửi mã OTP".<br>4. Hệ thống kiểm tra tính hợp lệ của thông tin vừa nhập (điền đủ, đúng định dạng) và kiểm tra tần suất gửi yêu cầu từ IP này để phòng chống spam.<br>5. Hệ thống kiểm tra Email. Nếu Email chưa từng được sử dụng, hệ thống sẽ gửi một mã OTP gồm 6 chữ số đến hòm thư của người dùng.<br>6. Người dùng mở Email, lấy mã OTP, nhập vào hệ thống và nhấn "Tạo tài khoản".<br>7. Hệ thống kiểm tra tính hợp lệ của mã OTP. Nếu chính xác, hệ thống khởi tạo tài khoản mới.<br>8. [cite_start]Hệ thống thông báo đăng ký thành công, tự động đăng nhập và đưa người dùng về Trang chủ. [cite: 966] |
| **Alternative Flow** | 3a. Người dùng chọn lệnh "Chuyển sang Đăng nhập". [cite_start]Hệ thống chuyển đổi biểu mẫu sang màn hình Đăng nhập. [cite: 966] |
| **Exception Flow** | 4a. Thiếu thông tin hoặc sai định dạng (Validation): Hệ thống hiển thị thông báo lỗi (màu đỏ) ngay tại các ô nhập liệu tương ứng và yêu cầu nhập lại.<br>4b. Gửi quá nhiều yêu cầu (Rate Limiting): Hệ thống chặn hành động và hiển thị thông báo "Bạn đã thao tác quá nhiều lần, vui lòng thử lại sau 10 phút".<br>5a. Email đã tồn tại: Hệ thống thông báo "Email này đã được đăng ký tài khoản".<br>7a. [cite_start]Mã OTP sai hoặc hết hạn: Hệ thống hiển thị cảnh báo "Mã OTP không chính xác hoặc đã hết hạn". [cite: 966] |
| **Business Rules** | - [cite_start]BR01-1: Mã OTP chỉ bao gồm 6 chữ số ngẫu nhiên và có thời gian hiệu lực giới hạn.<br>- BR01-2: Mật khẩu phải đáp ứng độ dài tối thiểu và độ phức tạp an toàn. [cite: 966] |
| **Non-Functional** | - [cite_start]NFR01-1: Hệ thống gửi Email chứa mã OTP trong thời gian không quá 5 giây kể từ lúc nhấn nút. [cite: 966] |

#### 1.2. [cite_start]Sequence diagram [cite: 967]
[cite_start]*(Sơ đồ tuần tự được đính kèm trong tài liệu gốc)* [cite: 967]

---

### [cite_start]2. Chức năng Đăng nhập (Login) [cite: 968]
#### 2.1. [cite_start]Đặc tả UC02: Đăng nhập (Login) [cite: 969]

| Trường | Nội dung |
| :--- | :--- |
| **Use Case ID** | [cite_start]UC02 [cite: 970] |
| **Use Case Name** | [cite_start]Đăng nhập [cite: 970] |
| **Description** | Là một Người dùng, tôi muốn đăng nhập vào hệ thống bằng Email và Mật khẩu. [cite_start]Hệ thống sẽ tự động nhận diện phân quyền và điều hướng tôi tới không gian làm việc phù hợp. [cite: 970] |
| **Actor(s)** | [cite_start]Người dùng (User), Quản trị viên (Admin) [cite: 970] |
| **Priority** | [cite_start]Must Have [cite: 970] |
| **Trigger** | [cite_start]Người dùng truy cập trang Đăng nhập và điền thông tin. [cite: 970] |
| **Pre-Condition(s)** | [cite_start]Người dùng đã có tài khoản hợp lệ trên hệ thống. [cite: 970] |
| **Post-Condition(s)** | [cite_start]Đăng nhập thành công, hệ thống thiết lập phiên làm việc và điều hướng người dùng tới giao diện tương ứng với quyền hạn. [cite: 970] |
| **Basic Flow** | 1. Người dùng truy cập trang Đăng nhập.<br>2. Người dùng nhập Email, Mật khẩu đã đăng ký và nhấn nút "Đăng nhập".<br>3. Hệ thống kiểm tra dữ liệu đầu vào (không được bỏ trống) và kiểm tra tần suất đăng nhập sai để ngăn chặn hành vi dò rỉ mật khẩu (Brute-force).<br>4. Hệ thống đối chiếu Email và Mật khẩu với cơ sở dữ liệu.<br>5. Nếu thông tin chính xác, hệ thống thiết lập phiên làm việc bảo mật cho người dùng.<br>6. [cite_start]Hệ thống tự động kiểm tra vai trò (Role) của tài khoản và tiến hành điều hướng:<br>- Nếu là User: Chuyển hướng sang trang Hồ sơ cá nhân (/user/profile).<br>- Nếu là Admin: Chuyển hướng sang trang Quản trị (/admin/profile). [cite: 970] |
| **Alternative Flow** | [cite_start]Không có. [cite: 970] |
| **Exception Flow** | 3a. Thiếu thông tin (Validation): Hệ thống báo lỗi yêu cầu điền đầy đủ Email và Mật khẩu.<br>3b. Đăng nhập sai quá nhiều lần (Rate Limiting): Hệ thống hiển thị thông báo "Tài khoản tạm khóa do đăng nhập sai nhiều lần. Vui lòng thử lại sau 15 phút" để bảo vệ tài khoản.<br>4a. [cite_start]Thông tin không khớp: Hệ thống báo lỗi "Email hoặc mật khẩu không chính xác" và yêu cầu người dùng nhập lại. [cite: 970] |
| **Business Rules** | - [cite_start]BR02-1: Quyền hạn điều hướng trang (URL trả về) phụ thuộc tuyệt đối vào cấp độ định danh của người dùng lưu trong hệ thống, người dùng không thể tự can thiệp. [cite: 970] |
| **Non-Functional** | - [cite_start]NFR02-1: Về mặt kỹ thuật, "Phiên làm việc" phải được quản lý bằng chuẩn mã hóa JSON Web Token (JWT).<br>- NFR02-2: Mật khẩu của người dùng bắt buộc phải được mã hóa bằng BCrypt khi đối chiếu. [cite: 970] |

#### 2.2. [cite_start]Sequence diagram [cite: 971]
[cite_start]*(Sơ đồ tuần tự được đính kèm trong tài liệu gốc)* [cite: 971]

---

### [cite_start]3. Chức năng Quên mật khẩu (Forgot Password) [cite: 972]
#### 3.1. [cite_start]Đặc tả UC03: Quên mật khẩu (Forgot Password) [cite: 973]

| Trường | Nội dung |
| :--- | :--- |
| **Use Case ID** | [cite_start]UC03 [cite: 974] |
| **Use Case Name** | [cite_start]Đổi / Quên mật khẩu [cite: 974] |
| **Description** | [cite_start]Là một Người dùng, tôi muốn thiết lập lại mật khẩu khi bị quên hoặc chủ động đổi mật khẩu để bảo vệ an toàn cho tài khoản cá nhân. [cite: 974] |
| **Actor(s)** | [cite_start]Người dùng (User), Quản trị viên (Admin) [cite: 974] |
| **Priority** | [cite_start]Should Have [cite: 974] |
| **Trigger** | [cite_start]Người dùng nhấn vào liên kết "Quên mật khẩu" ở màn hình Đăng nhập, hoặc chọn "Đổi mật khẩu" trong Quản lý tài khoản. [cite: 974] |
| **Pre-Condition(s)** | [cite_start]Người dùng phải truy cập được vào hòm thư Email đã đăng ký. [cite: 974] |
| **Post-Condition(s)** | [cite_start]Mật khẩu mới được cập nhật thành công vào cơ sở dữ liệu. [cite: 974] |
| **Basic Flow** | 1. Người dùng chọn lệnh "Quên mật khẩu" tại màn hình Đăng nhập.<br>2. Hệ thống yêu cầu cung cấp Email định danh.<br>3. Người dùng nhập Email và chọn lệnh "Gửi mã xác thực".<br>4. Hệ thống kiểm tra tần suất yêu cầu để chống spam, sau đó tra cứu thông tin và gửi mã OTP xác nhận về hòm thư Email.<br>5. Người dùng nhập mã OTP và nhập Mật khẩu mới mong muốn.<br>6. Hệ thống xác thực OTP. Nếu hợp lệ, hệ thống tiến hành mã hóa bảo mật mật khẩu mới và ghi đè lên dữ liệu cũ.<br>7. [cite_start]Hệ thống thông báo cập nhật thành công và đưa người dùng về lại trang Đăng nhập. [cite: 974] |
| **Alternative Flow** | [cite_start]Không có [cite: 974] |
| **Exception Flow** | 4a. Gửi quá nhiều yêu cầu (Rate Limiting): Hệ thống chặn lệnh gửi mã OTP và báo lỗi "Bạn đã thao tác quá nhiều lần. Vui lòng thử lại sau".<br>4b. [cite_start]Email không tồn tại: Hệ thống báo lỗi "Tài khoản Email không tồn tại" và chặn lệnh gửi mã. [cite: 974] |
| **Business Rules** | - [cite_start]BR03-1: Mật khẩu bắt buộc phải có độ dài tối thiểu 8 ký tự, có chữ số và ký tự đặc biệt để đảm bảo tiêu chuẩn an toàn. [cite: 974] |
| **Non-Functional** | - NFR03-1: Mật khẩu mới tuyệt đối không được lưu dưới dạng văn bản thô. [cite_start]Backend bắt buộc phải băm (hash) bằng thuật toán BCrypt trước khi lưu vào Database. [cite: 974] |

#### 3.2. [cite_start]Sequence diagram [cite: 975]
[cite_start]*(Sơ đồ tuần tự được đính kèm trong tài liệu gốc)* [cite: 975]

---

### [cite_start]4. Chức năng Cập nhật thông tin cá nhân (Edit Profile) [cite: 976]
#### 4.1. [cite_start]Đặc tả UC04: Cập nhật thông tin cá nhân (Edit Profile) [cite: 977]

| Trường | Nội dung |
| :--- | :--- |
| **Use Case ID** | [cite_start]UC04 [cite: 978] |
| **Use Case Name** | [cite_start]Cập nhật thông tin cá nhân [cite: 978] |
| **Description** | [cite_start]Là một Người dùng, tôi muốn cập nhật thông tin cá nhân của mình để hệ thống lưu trữ đúng dữ liệu liên lạc phục vụ cho quá trình xác thực và chăm sóc. [cite: 978] |
| **Actor(s)** | [cite_start]Người dùng (User) [cite: 978] |
| **Priority** | [cite_start]Must Have [cite: 978] |
| **Trigger** | [cite_start]Người dùng truy cập vào Menu "Tài khoản của tôi" và chọn tab "Hồ sơ cá nhân". [cite: 978] |
| **Pre-Condition(s)** | [cite_start]Người dùng đã đăng nhập vào hệ thống và đang sở hữu phiên làm việc (Token) còn hiệu lực. [cite: 978] |
| **Post-Condition(s)** | [cite_start]Thông tin cá nhân mới được cập nhật thành công vào cơ sở dữ liệu. [cite: 978] |
| **Basic Flow** | 1. Người dùng truy cập tab "Hồ sơ cá nhân".<br>2. Hệ thống xác thực danh tính người dùng thông qua phiên làm việc hiện tại.<br>3. Nếu hợp lệ, hệ thống truy xuất dữ liệu cá nhân tương ứng và hiển thị trên biểu mẫu (Form).<br>4. Người dùng thực hiện chỉnh sửa các trường thông tin mong muốn (Họ tên, Số điện thoại).<br>5. Người dùng nhấn nút "Lưu thay đổi".<br>6. Hệ thống kiểm tra tính hợp lệ của dữ liệu đầu vào.<br>7. Hệ thống ghi nhận thông tin mới vào cơ sở dữ liệu.<br>8. [cite_start]Hệ thống hiển thị thông báo "Cập nhật thành công" và làm mới lại dữ liệu hiển thị. [cite: 978] |
| **Alternative Flow** | [cite_start]Không có. [cite: 978] |
| **Exception Flow** | 2a. Lỗi xác thực (Authentication): Nếu phiên làm việc (Token) không hợp lệ, bị giả mạo hoặc đã hết hạn, hệ thống sẽ từ chối truy cập, trả về lỗi "Unauthorized" và yêu cầu đăng nhập lại.<br>6a. Thiếu thông tin (Validation): Người dùng bỏ trống trường thông tin bắt buộc (Họ tên) hoặc sai định dạng. [cite_start]Hệ thống hiển thị thông báo lỗi bôi đỏ tại trường tương ứng và chặn lệnh lưu. [cite: 978] |
| **Business Rules** | - BR04-1: Người dùng không được phép thay đổi Email đăng nhập. [cite_start]Trường Email được đặt ở chế độ Read-only (Chỉ đọc) trên giao diện. [cite: 978] |
| **Non-Functional** | [cite_start]*(Để trống trong tài liệu gốc)* [cite: 978] |

#### 4.2. [cite_start]Sequence diagram [cite: 979]
[cite_start]*(Sơ đồ tuần tự được đính kèm trong tài liệu gốc)* [cite: 979]

---

### [cite_start]Link Github [cite: 980]
* **1. [cite_start]Link Github nhóm:** [https://github.com/1105trxc/BT02_Nhom01](https://github.com/1105trxc/BT02_Nhom01) [cite: 981]
* **2. [cite_start]Link Github cá nhân:** [https://github.com/1105trxc/BT02_CaNhan](https://github.com/1105trxc/BT02_CaNhan) [cite: 982]
