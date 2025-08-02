import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider, CheckButtons, RadioButtons
import json
import os
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox, simpledialog

class InteractiveTemplateCreator:
    def __init__(self):
        self.grid_size = 8
        self.templates = {}
        self.current_digit = 0
        self.current_template_name = "default"
        self.current_grid = np.zeros((8, 8))
        self.drawing_mode = True  # True for drawing (1), False for erasing (-1)
        
        # Template transformation parameters
        self.template_params = {}
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """GUI 설정"""
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 6))
        self.fig.suptitle('Interactive Binary Digit Template Creator', fontsize=16)
        
        # Grid drawing area
        self.ax_grid = self.axes[0]
        self.ax_grid.set_title('Template Drawing Area (8x8)')
        self.ax_grid.set_xlim(-0.5, 7.5)
        self.ax_grid.set_ylim(-0.5, 7.5)
        self.ax_grid.set_aspect('equal')
        self.ax_grid.invert_yaxis()
        
        # Draw grid
        for i in range(9):
            self.ax_grid.axhline(i-0.5, color='lightgray', linewidth=0.5)
            self.ax_grid.axvline(i-0.5, color='lightgray', linewidth=0.5)
        
        # Initialize grid display
        self.grid_patches = []
        for i in range(8):
            row = []
            for j in range(8):
                rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                       facecolor='white', edgecolor='black', linewidth=0.5)
                self.ax_grid.add_patch(rect)
                row.append(rect)
            self.grid_patches.append(row)
        
        # Template list area
        self.ax_list = self.axes[1]
        self.ax_list.set_title('Saved Templates')
        self.ax_list.axis('off')
        
        # Preview area
        self.ax_preview = self.axes[2]
        self.ax_preview.set_title('Template Preview')
        self.ax_preview.axis('off')
        
        # Add buttons and controls
        self.add_controls()
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        
        self.update_display()
        
    def add_controls(self):
        """컨트롤 버튼들 추가"""
        # Clear button
        ax_clear = plt.axes([0.02, 0.8, 0.1, 0.05])
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_clear.on_clicked(self.clear_grid)
        
        # Toggle drawing mode button
        ax_toggle = plt.axes([0.02, 0.7, 0.1, 0.05])
        self.btn_toggle = Button(ax_toggle, 'Draw Mode')
        self.btn_toggle.on_clicked(self.toggle_mode)
        
        # Save template button
        ax_save = plt.axes([0.02, 0.6, 0.1, 0.05])
        self.btn_save = Button(ax_save, 'Save Template')
        self.btn_save.on_clicked(self.save_template)
        
        # Load template button
        ax_load = plt.axes([0.02, 0.5, 0.1, 0.05])
        self.btn_load = Button(ax_load, 'Load Template')
        self.btn_load.on_clicked(self.load_template)
        
        # Set parameters button
        ax_params = plt.axes([0.02, 0.4, 0.1, 0.05])
        self.btn_params = Button(ax_params, 'Set Params')
        self.btn_params.on_clicked(self.set_parameters)
        
        # Generate dataset button
        ax_generate = plt.axes([0.02, 0.3, 0.1, 0.05])
        self.btn_generate = Button(ax_generate, 'Generate Data')
        self.btn_generate.on_clicked(self.generate_dataset)
        
        # Digit selector - 라디오버튼으로 변경
        ax_digit = plt.axes([0.4, 0.15, 0.12, 0.4])  # 크기 증가
        self.digit_labels = [f'Digit {i}' for i in range(10)]
        self.digit_radio = RadioButtons(ax_digit, self.digit_labels, active=0)
        self.digit_radio.on_clicked(self.select_digit_radio)
        
        # 라디오 버튼 스타일 조정
        for label in self.digit_radio.labels:
            label.set_fontsize(11)
        # circles 속성이 없는 경우를 대비해 try-except 사용
        try:
            for circle in self.digit_radio.circles:
                circle.set_radius(0.05)
        except AttributeError:
            # circles 속성이 없는 matplotlib 버전
            pass
        
    def on_click(self, event):
        """마우스 클릭 이벤트 처리"""
        if event.inaxes == self.ax_grid:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= x < 8 and 0 <= y < 8:
                self.current_grid[y, x] = 1 if self.drawing_mode else 0
                self.update_grid_display()
                
    def on_drag(self, event):
        """마우스 드래그 이벤트 처리"""
        if event.inaxes == self.ax_grid and event.button == 1:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= x < 8 and 0 <= y < 8:
                self.current_grid[y, x] = 1 if self.drawing_mode else 0
                self.update_grid_display()
                
    def clear_grid(self, event):
        """그리드 초기화"""
        self.current_grid = np.zeros((8, 8))
        self.update_grid_display()
        
    def toggle_mode(self, event):
        """그리기/지우기 모드 토글"""
        self.drawing_mode = not self.drawing_mode
        mode_text = 'Draw Mode' if self.drawing_mode else 'Erase Mode'
        self.btn_toggle.label.set_text(mode_text)
        plt.draw()
        
    def select_digit_radio(self, label):
        """라디오버튼으로 숫자 선택"""
        # 'Digit 1' → 1로 변환
        digit = int(label.split()[-1])
        self.current_digit = digit
        self.update_display()
        print(f"✅ Selected digit: {self.current_digit}")
        
    def save_template(self, event):
        """템플릿 저장"""
        # Get template name from user
        root = tk.Tk()
        root.withdraw()
        template_name = simpledialog.askstring("Template Name", 
                                               f"Enter name for digit {self.current_digit} template:")
        root.destroy()
        
        if template_name:
            if self.current_digit not in self.templates:
                self.templates[self.current_digit] = {}
            
            # Convert 0/1 to -1/1 for BNN
            bnn_template = np.where(self.current_grid == 1, 1, -1)
            self.templates[self.current_digit][template_name] = bnn_template.copy()
            
            # Initialize default parameters for this template
            self.init_template_params(self.current_digit, template_name)
            
            self.update_display()
            print(f"Template '{template_name}' saved for digit {self.current_digit}")
            
    def init_template_params(self, digit, template_name):
        """템플릿 기본 파라미터 초기화"""
        key = f"{digit}_{template_name}"
        self.template_params[key] = {
            'shift_prob': 0.5,
            'max_shift': 2,
            'noise_prob': 0.1,
            'noise_rate': 0.05,
            'deform_prob': 0.3,
            'rotation_prob': 0.2,
            'scale_prob': 0.1,
            'custom_transform': None
        }
        
    def load_template(self, event):
        """Load template"""
        if self.current_digit in self.templates and self.templates[self.current_digit]:
            # Show available templates for current digit
            template_names = list(self.templates[self.current_digit].keys())
            
            root = tk.Tk()
            root.withdraw()
            
            # Create selection dialog
            selection = simpledialog.askstring("Load Template", 
                                             f"Available templates for digit {self.current_digit}:\n" + 
                                             "\n".join(template_names) + 
                                             "\n\nEnter template name:")
            root.destroy()
            
            if selection and selection in template_names:
                # Convert -1/1 back to 0/1 for display
                loaded_template = self.templates[self.current_digit][selection]
                self.current_grid = np.where(loaded_template == 1, 1, 0)
                self.update_grid_display()
                print(f"Template '{selection}' loaded")
        else:
            print(f"No templates available for digit {self.current_digit}")
            
    def set_parameters(self, event):
        """템플릿 파라미터 설정"""
        if self.current_digit in self.templates and self.templates[self.current_digit]:
            template_names = list(self.templates[self.current_digit].keys())
            
            root = tk.Tk()
            root.withdraw()
            
            # Select template
            selection = simpledialog.askstring("Set Parameters", 
                                             f"Available templates for digit {self.current_digit}:\n" + 
                                             "\n".join(template_names) + 
                                             "\n\nEnter template name:")
            
            if selection and selection in template_names:
                self.open_parameter_window(self.current_digit, selection)
            
            root.destroy()
        else:
            print(f"No templates available for digit {self.current_digit}")
            
    def open_parameter_window(self, digit, template_name):
        """파라미터 설정 창 열기"""
        key = f"{digit}_{template_name}"
        params = self.template_params.get(key, {})
        
        # Create parameter window
        param_window = tk.Toplevel()
        param_window.title(f"Parameters for {digit}_{template_name}")
        param_window.geometry("400x500")
        
        # Parameter sliders
        param_vars = {}
        
        tk.Label(param_window, text=f"Parameters for Digit {digit} - {template_name}", 
                font=("Arial", 12, "bold")).pack(pady=10)
        
        # Shift parameters
        tk.Label(param_window, text="Shift Parameters", font=("Arial", 10, "bold")).pack()
        
        # Shift probability
        tk.Label(param_window, text="Shift Probability").pack()
        param_vars['shift_prob'] = tk.DoubleVar(value=params.get('shift_prob', 0.5))
        tk.Scale(param_window, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL,
                variable=param_vars['shift_prob']).pack()
        
        # Max shift
        tk.Label(param_window, text="Max Shift").pack()
        param_vars['max_shift'] = tk.IntVar(value=params.get('max_shift', 2))
        tk.Scale(param_window, from_=0, to=4, orient=tk.HORIZONTAL,
                variable=param_vars['max_shift']).pack()
        
        # Noise parameters
        tk.Label(param_window, text="Noise Parameters", font=("Arial", 10, "bold")).pack()
        
        tk.Label(param_window, text="Noise Probability").pack()
        param_vars['noise_prob'] = tk.DoubleVar(value=params.get('noise_prob', 0.1))
        tk.Scale(param_window, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                variable=param_vars['noise_prob']).pack()
        
        tk.Label(param_window, text="Noise Rate").pack()
        param_vars['noise_rate'] = tk.DoubleVar(value=params.get('noise_rate', 0.05))
        tk.Scale(param_window, from_=0.0, to=0.3, resolution=0.01, orient=tk.HORIZONTAL,
                variable=param_vars['noise_rate']).pack()
        
        # Deformation parameters
        tk.Label(param_window, text="Deformation Probability", font=("Arial", 10, "bold")).pack()
        param_vars['deform_prob'] = tk.DoubleVar(value=params.get('deform_prob', 0.3))
        tk.Scale(param_window, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL,
                variable=param_vars['deform_prob']).pack()
        
        # Save button
        def save_params():
            for param_name, var in param_vars.items():
                self.template_params[key][param_name] = var.get()
            print(f"Parameters saved for {key}")
            param_window.destroy()
            
        tk.Button(param_window, text="Save Parameters", command=save_params).pack(pady=20)
        
    def update_grid_display(self):
        """그리드 디스플레이 업데이트"""
        for i in range(8):
            for j in range(8):
                color = 'black' if self.current_grid[i, j] == 1 else 'white'
                self.grid_patches[i][j].set_facecolor(color)
        plt.draw()
        
    def update_display(self):
        """전체 디스플레이 업데이트"""
        self.update_grid_display()
        self.update_template_list()
        
    def update_template_list(self):
        """템플릿 리스트 업데이트"""
        self.ax_list.clear()
        self.ax_list.set_title(f'Templates for Digit {self.current_digit}')
        self.ax_list.axis('off')
        
        if self.current_digit in self.templates:
            template_names = list(self.templates[self.current_digit].keys())
            y_pos = 0.9
            for name in template_names:
                self.ax_list.text(0.1, y_pos, f"• {name}", fontsize=10, transform=self.ax_list.transAxes)
                y_pos -= 0.1
        else:
            self.ax_list.text(0.1, 0.5, "No templates saved", fontsize=10, transform=self.ax_list.transAxes)
            
        plt.draw()
        
    def generate_dataset(self, event):
        """데이터셋 생성"""
        if not self.templates:
            print("No templates available. Please create some templates first.")
            return
            
        # Get parameters from user
        root = tk.Tk()
        root.withdraw()
        
        samples_per_digit = simpledialog.askinteger("Dataset Generation", 
                                                   "Samples per digit:", initialvalue=500)
        if not samples_per_digit:
            root.destroy()
            return
            
        test_size = simpledialog.askfloat("Dataset Generation", 
                                         "Test size (0.0-1.0):", initialvalue=0.2)
        if test_size is None:
            root.destroy()
            return
            
        root.destroy()
        
        print("Generating dataset...")
        generator = BinaryDigitDataGenerator(self.templates, self.template_params)
        X_train, X_test, y_train, y_test = generator.create_dataset(
            samples_per_digit=samples_per_digit, 
            test_size=test_size
        )
        
        print(f"Dataset generated successfully!")
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Save to files
        np.save('X_train.npy', X_train)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)
        print("Dataset saved to .npy files")
        
        return X_train, X_test, y_train, y_test
        
    def save_templates(self, filename="templates.json"):
        """템플릿을 파일로 저장"""
        # Convert numpy arrays to lists for JSON serialization
        save_data = {
            'templates': {},
            'params': self.template_params
        }
        
        for digit, digit_templates in self.templates.items():
            save_data['templates'][str(digit)] = {}
            for name, template in digit_templates.items():
                save_data['templates'][str(digit)][name] = template.tolist()
                
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"Templates saved to {filename}")
        
    def load_templates(self, filename="templates.json"):
        """파일에서 템플릿 로드"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                save_data = json.load(f)
                
            # Convert back to numpy arrays
            self.templates = {}
            for digit_str, digit_templates in save_data['templates'].items():
                digit = int(digit_str)
                self.templates[digit] = {}
                for name, template_list in digit_templates.items():
                    self.templates[digit][name] = np.array(template_list)
                    
            self.template_params = save_data.get('params', {})
            self.update_display()
            print(f"Templates loaded from {filename}")
        else:
            print(f"File {filename} not found")


class BinaryDigitDataGenerator:
    def __init__(self, templates, template_params):
        self.templates = templates
        self.template_params = template_params
        
    def shift_image(self, img, dx=0, dy=0):
        """이미지 이동"""
        result = np.full_like(img, -1)
        src_x_start = max(0, -dx)
        src_x_end = min(8, 8 - dx)
        src_y_start = max(0, -dy)
        src_y_end = min(8, 8 - dy)
        dst_x_start = max(0, dx)
        dst_y_start = max(0, dy)
        
        if src_x_end > src_x_start and src_y_end > src_y_start:
            result[dst_y_start:dst_y_start+(src_y_end-src_y_start),
                   dst_x_start:dst_x_start+(src_x_end-src_x_start)] = \
                img[src_y_start:src_y_end, src_x_start:src_x_end]
        return result

    def add_noise(self, img, noise_rate=0.05):
        """노이즈 추가"""
        result = img.copy()
        mask = np.random.random(img.shape) < noise_rate
        result[mask] = -result[mask]
        return result

    def slight_deform(self, img):
        """약간의 변형"""
        result = img.copy()
        num_changes = np.random.choice([1, 2, 3])
        
        for _ in range(num_changes):
            y, x = np.random.randint(1, 7), np.random.randint(1, 7)
            neighbors = []
            
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < 8 and 0 <= nx < 8:
                        neighbors.append(result[ny, nx])
            
            if len(neighbors) > 0:
                unique, counts = np.unique(neighbors, return_counts=True)
                majority_value = unique[np.argmax(counts)]
                if np.random.random() < 0.6:  # 60% chance to follow majority
                    result[y, x] = majority_value
                    
        return result

    def apply_transformations(self, template, digit, template_name):
        """템플릿별 변형 적용"""
        key = f"{digit}_{template_name}"
        params = self.template_params.get(key, {})
        
        result = template.copy()
        
        # Shift transformation
        if np.random.random() < params.get('shift_prob', 0.5):
            max_shift = params.get('max_shift', 2)
            dx = np.random.randint(-max_shift, max_shift + 1)
            dy = np.random.randint(-max_shift, max_shift + 1)
            result = self.shift_image(result, dx, dy)
        
        # Noise transformation
        if np.random.random() < params.get('noise_prob', 0.1):
            noise_rate = params.get('noise_rate', 0.05)
            result = self.add_noise(result, noise_rate)
        
        # Deformation
        if np.random.random() < params.get('deform_prob', 0.3):
            result = self.slight_deform(result)
            
        return result

    def generate_samples(self, digit, num_samples):
        """특정 숫자의 샘플 생성"""
        if digit not in self.templates:
            print(f"No templates for digit {digit}")
            return [], []
            
        templates = self.templates[digit]
        template_names = list(templates.keys())
        
        samples = []
        labels = []
        
        for _ in range(num_samples):
            # Randomly select template
            template_name = np.random.choice(template_names)
            base_template = templates[template_name]
            
            # Apply transformations
            transformed = self.apply_transformations(base_template, digit, template_name)
            
            samples.append(transformed.flatten())
            labels.append(digit)
            
        return samples, labels

    def create_dataset(self, samples_per_digit=500, test_size=0.2, random_state=42):
        """전체 데이터셋 생성"""
        all_samples = []
        all_labels = []
        
        print("Generating dataset...")
        
        for digit in range(10):
            if digit in self.templates:
                print(f"Generating {samples_per_digit} samples for digit {digit}...")
                samples, labels = self.generate_samples(digit, samples_per_digit)
                all_samples.extend(samples)
                all_labels.extend(labels)
            else:
                print(f"Skipping digit {digit} - no templates available")
        
        if not all_samples:
            raise ValueError("No samples generated. Please create templates first.")
            
        X = np.array(all_samples, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Dataset created: {len(X)} total samples")
        print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test


def main():
    """메인 함수"""
    print("Starting Interactive Template Creator...")
    print("Instructions:")
    print("1. Select a digit (0-9) from the checkboxes")
    print("2. Draw templates by clicking/dragging on the 8x8 grid")
    print("3. Use 'Draw Mode'/'Erase Mode' button to toggle between drawing and erasing")
    print("4. Save templates with meaningful names")
    print("5. Set custom parameters for each template")
    print("6. Generate dataset when ready")
    print("\nControls:")
    print("- Left click/drag: Draw or erase pixels")
    print("- Clear: Reset current grid")
    print("- Toggle Mode: Switch between draw/erase")
    print("- Save Template: Save current grid as template")
    print("- Load Template: Load existing template")
    print("- Set Params: Customize transformation parameters")
    print("- Generate Data: Create final dataset")
    
    creator = InteractiveTemplateCreator()
    
    # Try to load existing templates
    creator.load_templates()
    
    plt.show()
    
    # Save templates when closing
    creator.save_templates()

if __name__ == "__main__":
    main() 