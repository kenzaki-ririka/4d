import numpy as np
import numpy as numpy_cpu  # 保留原始NumPy用于可视化
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons

# GPU 加速：尝试使用 CuPy，如果不可用则回退到 NumPy
try:
    import cupy as cp
    np = cp  # 使用 GPU
    USE_GPU = True
    print("✓ 使用 CuPy GPU 加速")
except ImportError:
    # 回退到 CPU
    USE_GPU = False
    print("✗ CuPy 不可用，使用 NumPy CPU 计算")

def to_numpy(arr):
    """将数组转换为NumPy数组（用于matplotlib可视化）"""
    if USE_GPU:
        return arr.get()  # CuPy -> NumPy
    return arr  # 已经是NumPy

# --- 4D 流体模拟核心类 ---
class FluidSimulator4D:
    def __init__(self, N=12, L=2*np.pi, nu=0.005, dt=0.02):
        self.N = N
        self.L = L
        self.nu = nu
        self.dt = dt
        
        k = 2 * np.pi / L * np.fft.fftfreq(N, d=1/N)
        self.K = np.array(np.meshgrid(k, k, k, k, indexing='ij'))
        self.K2 = np.sum(self.K**2, axis=0)
        self.K2[0,0,0,0] = 1e-10
        self.k_max = np.max(k) * 2/3
        self.dealias_mask = self.K2 < (self.k_max**2)
        x = np.linspace(0, L, N, endpoint=False)
        self.X_grid = np.array(np.meshgrid(x, x, x, x, indexing='ij'))
        self.u_hat = np.zeros((4, N, N, N, N), dtype=complex)
        self.time = 0.0  # 模拟时间
        
        # 壁面设置: 'none', 'damping', 'mirror'
        self.wall_mode = 'none'
        self.wall_mask = self._create_wall_mask()
        
        self.setup_initial_condition('TurbulentPour')  # 默认使用带扰动的倒水模式
    
    def _create_wall_mask(self):
        """创建壁面阻尼mask，边界附近为0，内部为1，使用平滑过渡"""
        N, L = self.N, self.L
        
        def dim_mask(coord):
            t = coord / L
            return np.sin(np.pi * t) ** 2
        
        mask = np.ones((N, N, N, N))
        for i, grid in enumerate(self.X_grid):
            mask *= dim_mask(grid)
        
        return mask
    
    def _apply_mirror_bc(self):
        """应用镜像边界条件（自由滑移）
        
        原理：
        - 法向速度 u_i 在第i维的边界(索引0和N-1)设为0
        - 通过反对称化边界附近的值来实现
        """
        u_phys = np.fft.ifftn(self.u_hat, axes=(1,2,3,4)).real
        N = self.N
        
        for dim in range(4):
            # 对于第dim维的速度分量，在该维度的边界强制为0
            # 使用切片来操作正确的轴
            
            # 构建切片索引
            slicer_0 = [slice(None)] * 5  # [component, x, y, z, w]
            slicer_0[0] = dim  # 选择第dim个速度分量
            slicer_0[dim + 1] = 0  # 选择该维度的第0个索引
            
            slicer_end = [slice(None)] * 5
            slicer_end[0] = dim
            slicer_end[dim + 1] = N - 1  # 选择该维度的最后一个索引
            
            # 设置边界法向速度为0
            u_phys[tuple(slicer_0)] = 0
            u_phys[tuple(slicer_end)] = 0
            
            # 可选：使边界附近的法向速度反对称（增强反弹效果）
            slicer_1 = [slice(None)] * 5
            slicer_1[0] = dim
            slicer_1[dim + 1] = 1
            
            slicer_n2 = [slice(None)] * 5
            slicer_n2[0] = dim
            slicer_n2[dim + 1] = N - 2
            
            # 轻微衰减边界附近的法向速度
            u_phys[tuple(slicer_1)] *= 0.8
            u_phys[tuple(slicer_n2)] *= 0.8
        
        self.u_hat = np.fft.fftn(u_phys, axes=(1,2,3,4)) 

    def setup_initial_condition(self, mode):
        N, L = self.N, self.L
        X, Y, Z, W = self.X_grid[0], self.X_grid[1], self.X_grid[2], self.X_grid[3]
        u_phys = np.zeros((4, N, N, N, N))
        
        if mode == 'Random':
            u_phys = np.random.randn(4, N, N, N, N) * 0.5
        elif mode == 'VortexTube':
            # 涡管：XY平面上的旋转流，沿W方向调制
            r2 = (X - L/2)**2 + (Y - L/2)**2
            u_rot = np.exp(-r2 * 0.5) * 2.0  # 减小衰减因子使涡管更大，增大强度
            modulation = np.sin(2 * np.pi * W / L)
            u_phys[0] = -(Y - L/2) * u_rot * modulation
            u_phys[1] =  (X - L/2) * u_rot * modulation
        elif mode == 'DoubleRotation':
            u_phys[0] = -np.sin(Y); u_phys[1] =  np.sin(X)
            u_phys[2] = -np.sin(W); u_phys[3] =  np.sin(Z)
        elif mode == 'PouringStream':
            # 超长方体液柱，与XY平面平行
            # 可视化显示的是W方向速度（颜色）和XY方向速度（箭头）
            
            # 可配置参数：
            stream_x_center = L / 2      # X方向中心
            stream_y_center = L / 2      # Y方向中心
            stream_x_width = L * 0.25    # X方向宽度
            stream_y_width = L * 0.25    # Y方向宽度
            stream_z_center = L / 2      # Z方向中心
            stream_z_width = L * 0.4     # Z方向宽度
            # W方向：液柱只占据上半部分，这样可以观察"正前方"的液体从静止变为流动
            stream_w_start = L * 0.6     # W方向起始位置（上部）
            stream_w_end = L            # W方向结束位置
            stream_velocity = -2       # 速度（负W方向，向下流动）
            
            # 创建超长方体区域的mask
            mask_x = np.abs(X - stream_x_center) < stream_x_width / 2
            mask_y = np.abs(Y - stream_y_center) < stream_y_width / 2
            mask_z = np.abs(Z - stream_z_center) < stream_z_width / 2
            mask_w = (W >= stream_w_start) & (W <= stream_w_end)  # W方向只占据上部
            stream_mask = mask_x & mask_y & mask_z & mask_w
            
            # 设置W方向的速度
            u_phys[3] = stream_velocity * stream_mask.astype(float)
            
        elif mode == 'TurbulentPour':
            # 带随机扰动的倒水模式 - 更容易产生湍流
            stream_x_center = L / 2
            stream_y_center = L / 2
            stream_x_width = L * 0.3
            stream_y_width = L * 0.3
            stream_z_center = L / 2
            stream_z_width = L * 0.5
            stream_w_start = L * 0.5
            stream_w_end = L
            stream_velocity = -2.5
            
            mask_x = np.abs(X - stream_x_center) < stream_x_width / 2
            mask_y = np.abs(Y - stream_y_center) < stream_y_width / 2
            mask_z = np.abs(Z - stream_z_center) < stream_z_width / 2
            mask_w = (W >= stream_w_start) & (W <= stream_w_end)
            stream_mask = mask_x & mask_y & mask_z & mask_w
            mask_float = stream_mask.astype(float)
            
            # 主流速度（W方向）
            u_phys[3] = stream_velocity * mask_float
            
            # 添加随机扰动到所有方向，促进湍流发展
            perturbation_strength = 0.5  # 扰动强度
            u_phys[0] += perturbation_strength * np.random.randn(N, N, N, N) * mask_float
            u_phys[1] += perturbation_strength * np.random.randn(N, N, N, N) * mask_float
            u_phys[2] += perturbation_strength * np.random.randn(N, N, N, N) * mask_float
            u_phys[3] += perturbation_strength * 0.3 * np.random.randn(N, N, N, N) * mask_float
            
        self.u_hat = np.fft.fftn(u_phys, axes=(1,2,3,4))
        self.project_pressure()

    def project_pressure(self):
        div_u = np.sum(self.K * self.u_hat, axis=0)
        for i in range(4):
            self.u_hat[i] -= self.K[i] * div_u / self.K2

    def compute_nonlinear(self):
        u_phys = np.fft.ifftn(self.u_hat, axes=(1,2,3,4)).real
        convection = np.zeros_like(u_phys)
        for i in range(4):
            grad_ui = np.zeros((4, self.N, self.N, self.N, self.N))
            for j in range(4):
                deriv_hat = 1j * self.K[j] * self.u_hat[i]
                grad_ui[j] = np.fft.ifftn(deriv_hat).real 
            convection[i] = np.sum(u_phys * grad_ui, axis=0)
        return -np.fft.fftn(convection, axes=(1,2,3,4))

    def step(self):
        nonlinear = self.compute_nonlinear() * self.dealias_mask
        self.u_hat += nonlinear * self.dt
        self.project_pressure()
        self.u_hat *= np.exp(-self.nu * self.K2 * self.dt)
        
        # 应用壁面边界条件
        if self.wall_mode == 'damping':
            u_phys = np.fft.ifftn(self.u_hat, axes=(1,2,3,4)).real
            u_phys *= self.wall_mask  # 边界附近速度衰减为0
            self.u_hat = np.fft.fftn(u_phys, axes=(1,2,3,4))
        elif self.wall_mode == 'mirror':
            self._apply_mirror_bc()
        # 'none' 模式不做任何处理（周期边界）
        
        self.time += self.dt

# --- 升级版 GUI 类 (支持可切换可视化平面) ---
class FluidGUI:
    def __init__(self, simulator):
        self.sim = simulator
        self.running = False  # 启动时暂停，需要手动点击开始
        self.auto_scale = False
        self.fixed_scale_val = 0.8
        self.color_component = 3   # 默认显示W分量 (0=X, 1=Y, 2=Z, 3=W)
        
        # 可视化平面设置：plane_axes = (水平轴, 垂直轴)，另外两个维度用切片
        # 维度索引：0=X, 1=Y, 2=Z, 3=W
        self.dim_names = ['X', 'Y', 'Z', 'W']
        self.plane_axes = (0, 1)  # 默认显示XY平面
        self.slice_dims = [2, 3]  # 切片维度
        self.slice_idx = [self.sim.N // 2, self.sim.N // 2]  # 切片索引
        
        self.fig, self.ax = plt.subplots(figsize=(11, 7))
        plt.subplots_adjust(left=0.08, bottom=0.35, right=0.78)
        
        # 初始化图像 (使用CPU NumPy用于matplotlib)
        self.im = self.ax.imshow(numpy_cpu.zeros((self.sim.N, self.sim.N)), 
                                 extent=[0, self.sim.L, 0, self.sim.L], 
                                 origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
        self.quiver = None 
        self.title = self.ax.set_title(f"4D Simulation (N={self.sim.N}) | t = {self.sim.time:.2f}")
        self.update_axis_labels()
        
        self.cbar = self.fig.colorbar(self.im, ax=self.ax)
        self.component_labels = ['$u_x$', '$u_y$', '$u_z$', '$u_w$']
        self.cbar.set_label(f"Velocity {self.component_labels[self.color_component]}")

        # --- 控件布局 ---
        # 1. 切片滑块（动态标签）
        ax_slice1 = plt.axes([0.15, 0.2, 0.50, 0.03])
        ax_slice2 = plt.axes([0.15, 0.15, 0.50, 0.03])
        self.slider_slice1 = Slider(ax_slice1, f'{self.dim_names[self.slice_dims[0]]}-Slice', 
                                    0, self.sim.N-1, valinit=self.slice_idx[0], valstep=1)
        self.slider_slice2 = Slider(ax_slice2, f'{self.dim_names[self.slice_dims[1]]}-Slice', 
                                    0, self.sim.N-1, valinit=self.slice_idx[1], valstep=1)
        
        # 2. 初始条件单选框
        ax_radio = plt.axes([0.02, 0.42, 0.12, 0.18])
        self.radio = RadioButtons(ax_radio, ('TurbulentPour', 'PouringStream', 'Random', 'VortexTube', 'DoubleRotation'))
        
        # 3. 可视化平面选择器
        ax_plane = plt.axes([0.02, 0.6, 0.12, 0.25])
        self.plane_radio = RadioButtons(ax_plane, ('XY', 'XZ', 'XW', 'YZ', 'YW', 'ZW'))
        
        # 4. 暂停按钮
        ax_btn = plt.axes([0.80, 0.15, 0.1, 0.04])
        self.btn_pause = Button(ax_btn, 'Pause/Run')
        
        # 5. 自动缩放复选框
        ax_check = plt.axes([0.80, 0.22, 0.12, 0.08])
        self.check = CheckButtons(ax_check, ['Auto Scale'], [False])
        
        # 6. 速度分量选择器
        ax_vel_radio = plt.axes([0.80, 0.35, 0.12, 0.12])
        self.vel_radio = RadioButtons(ax_vel_radio, ('uw (4D)', 'uz', 'uy', 'ux'))
        
        # 7. 壁面模式选择器
        ax_walls = plt.axes([0.80, 0.52, 0.12, 0.12])
        self.walls_radio = RadioButtons(ax_walls, ('None', 'Damping', 'Mirror'))
        
        # --- 事件绑定 ---
        self.slider_slice1.on_changed(self.update_slice_idx)
        self.slider_slice2.on_changed(self.update_slice_idx)
        self.radio.on_clicked(self.change_init_mode)
        self.plane_radio.on_clicked(self.change_plane)
        self.btn_pause.on_clicked(self.toggle_run)
        self.check.on_clicked(self.toggle_scale_mode)
        self.vel_radio.on_clicked(self.change_color_component)
        self.walls_radio.on_clicked(self.change_wall_mode)

        # 定时器
        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self.on_timer)
        self.timer.start()

    def update_axis_labels(self):
        """更新坐标轴标签"""
        self.ax.set_xlabel(self.dim_names[self.plane_axes[0]])
        self.ax.set_ylabel(self.dim_names[self.plane_axes[1]])

    def change_plane(self, label):
        """切换可视化平面"""
        plane_map = {
            'XY': (0, 1), 'XZ': (0, 2), 'XW': (0, 3),
            'YZ': (1, 2), 'YW': (1, 3), 'ZW': (2, 3)
        }
        self.plane_axes = plane_map.get(label, (0, 1))
        
        # 计算新的切片维度（不在平面内的维度）
        all_dims = {0, 1, 2, 3}
        self.slice_dims = sorted(all_dims - set(self.plane_axes))
        
        # 更新滑块标签
        self.slider_slice1.label.set_text(f'{self.dim_names[self.slice_dims[0]]}-Slice')
        self.slider_slice2.label.set_text(f'{self.dim_names[self.slice_dims[1]]}-Slice')
        
        # 更新坐标轴标签
        self.update_axis_labels()
        
        # 重置箭头（因为方向改变了）
        if self.quiver is not None:
            self.quiver.remove()
            self.quiver = None
        
        self.update_plot()

    def on_timer(self):
        if self.running:
            self.sim.step()
            self.update_plot()

    def update_slice_idx(self, val):
        self.slice_idx[0] = int(self.slider_slice1.val)
        self.slice_idx[1] = int(self.slider_slice2.val)
        self.update_plot()

    def change_init_mode(self, label):
        self.sim.time = 0.0
        self.sim.setup_initial_condition(label)
        self.update_plot()

    def toggle_run(self, event):
        self.running = not self.running

    def toggle_scale_mode(self, label):
        self.auto_scale = not self.auto_scale
    
    def change_wall_mode(self, label):
        """切换壁面模式"""
        mode_map = {'None': 'none', 'Damping': 'damping', 'Mirror': 'mirror'}
        self.sim.wall_mode = mode_map.get(label, 'none')
    
    def change_color_component(self, label):
        component_map = {'ux': 0, 'uy': 1, 'uz': 2, 'uw (4D)': 3}
        self.color_component = component_map.get(label, 3)
        self.cbar.set_label(f"Velocity {self.component_labels[self.color_component]}")
        self.update_plot()
        
    def update_plot(self):
        u_phys = np.fft.ifftn(self.sim.u_hat, axes=(1,2,3,4)).real
        
        # 根据当前平面和切片索引提取2D切片
        # u_phys shape: (4, N, N, N, N) 对应 (component, x, y, z, w)
        idx = [slice(None)] * 5  # 初始化为全部选择
        idx[0] = slice(None)  # 保留所有速度分量
        idx[self.slice_dims[0] + 1] = self.slice_idx[0]  # +1 因为第0维是分量
        idx[self.slice_dims[1] + 1] = self.slice_idx[1]
        
        u_slice = u_phys[tuple(idx)]  # shape: (4, N, N)
        
        # 需要确保正确的轴顺序：(component, axis0, axis1)
        # 根据plane_axes排列数据
        remaining_axes = [i for i in range(4) if i not in self.slice_dims]
        # remaining_axes 现在对应 plane_axes
        
        # 提取各分量
        u_components = [u_slice[i] for i in range(4)]
        
        # 颜色对应的速度分量
        u_color = u_components[self.color_component]
        
        # 箭头对应平面内两个方向的速度
        u_arrow_h = u_components[self.plane_axes[0]]  # 水平方向
        u_arrow_v = u_components[self.plane_axes[1]]  # 垂直方向
        
        # 转换为NumPy用于matplotlib
        u_color_np = to_numpy(u_color)
        u_arrow_h_np = to_numpy(u_arrow_h)
        u_arrow_v_np = to_numpy(u_arrow_v)
        
        # 缩放逻辑
        if self.auto_scale:
            v_max = max(float(numpy_cpu.max(numpy_cpu.abs(u_color_np))), 0.001)
        else:
            v_max = self.fixed_scale_val
            
        self.im.set_clim(-v_max, v_max)
        self.im.set_data(u_color_np.T) 
        
        step = 1
        if self.quiver is None:
            X, Y = numpy_cpu.meshgrid(numpy_cpu.linspace(0, self.sim.L, self.sim.N), 
                               numpy_cpu.linspace(0, self.sim.L, self.sim.N))
            self.quiver = self.ax.quiver(X[::step, ::step], Y[::step, ::step], 
                                         u_arrow_h_np.T[::step, ::step], u_arrow_v_np.T[::step, ::step],
                                         color='k', scale=20, pivot='mid')
        else:
            self.quiver.set_UVC(u_arrow_h_np.T[::step, ::step], u_arrow_v_np.T[::step, ::step])
        
        # 更新标题
        slice_info = f"{self.dim_names[self.slice_dims[0]]}={self.slice_idx[0]}, {self.dim_names[self.slice_dims[1]]}={self.slice_idx[1]}"
        self.title.set_text(f"4D Simulation | t = {self.sim.time:.2f} | {slice_info}")
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    # 使用稍微大一点的粘性 nu=0.01，这样你看固定模式时，衰减会比较明显
    print("启动模拟... 请尝试取消勾选右侧的 'Auto Scale' 来观察能量衰减。")
    sim = FluidSimulator4D(N=25, nu=0, dt=0.02)
    gui = FluidGUI(sim)
    plt.show()