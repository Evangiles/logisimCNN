import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def generate_im2col_hex(input_image, kernel_size, stride, filename):
    """
    8x8 입력 이미지(Numpy 배열)와 3x3 커널에 대한 im2col 데이터를 생성하고
    Logisim 회로에 맞는 .hex 파일로 저장합니다. (저장 순서: 열 우선 방식)
    """
    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image)
        
    image_h, image_w = input_image.shape
    kernel_h, kernel_w = kernel_size, kernel_size

    out_h = (image_h - kernel_h) // stride + 1
    out_w = (image_w - kernel_w) // stride + 1

    im2col_data = []
    # 열(Column)을 먼저 순회하여 데이터를 추출
    for j in range(out_w):  # x 좌표
        for i in range(out_h):  # y 좌표
            # 3x3 패치를 뽑아내서 1차원으로 펼침
            patch = input_image[i*stride : i*stride+kernel_h, j*stride : j*stride+kernel_w]
            im2col_data.extend(patch.flatten())

    # Logisim .hex 파일 형식으로 저장
    with open(filename, 'w') as f:
        f.write("v2.0 raw\n")
        # 9개 값(하나의 열)마다 줄바꿈하며 파일 작성
        for i in range(0, len(im2col_data), 9):
            chunk = im2col_data[i:i+9]
            hex_values = ' '.join(map(str, chunk))
            f.write(hex_values + ' \n')
    
    print(f"✅ '{filename}' 파일이 성공적으로 생성되었습니다. (총 {len(im2col_data)} 비트)")

class InteractiveHexGenerator:
    """
    matplotlib을 이용해 8x8 그리드에 그림을 그리고,
    im2col 변환을 거쳐 .hex 파일로 저장하는 GUI 클래스
    """
    def __init__(self, grid_size=8):
        self.grid_size = grid_size
        self.grid_data = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Matplotlib 그림판 설정
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(bottom=0.25) # 버튼을 위한 하단 공간 확보
        
        self.image = self.ax.imshow(self.grid_data, cmap='gray_r', vmin=0, vmax=1)
        
        # 그리드 모양 설정
        self.ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        self.ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
        self.ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        self.ax.set_title("Left-click/drag to draw, Right-click/drag to erase", fontsize=12)

        # 마우스 이벤트 연결 (클릭과 드래그 핸들러 분리)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # 버튼 생성
        ax_clear = plt.axes([0.5, 0.05, 0.18, 0.075])
        self.btn_clear = Button(ax_clear, 'Clear (지우기)')
        self.btn_clear.on_clicked(self.clear_grid)
        
        ax_save = plt.axes([0.7, 0.05, 0.2, 0.075])
        self.btn_save = Button(ax_save, 'Save HEX')
        self.btn_save.on_clicked(self.save_hex)

    def on_press(self, event):
        # Ignore if outside canvas or not a mouse button event
        if event.inaxes != self.ax or not event.button:
            return
        
        try: # Exception handling for coordinates outside image bounds
            x, y = int(round(event.xdata)), int(round(event.ydata))
        except (ValueError, TypeError):
            return

        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            if event.button == 1:  # Left click: draw (1)
                self.grid_data[y, x] = 1
            elif event.button == 3: # Right click: erase (0)
                self.grid_data[y, x] = 0
            self.update_canvas()

    def on_motion(self, event):
        # Ignore if outside canvas or no mouse button pressed
        if event.inaxes != self.ax or not event.buttons:
            return
            
        try: # Exception handling for coordinates outside image bounds
            x, y = int(round(event.xdata)), int(round(event.ydata))
        except (ValueError, TypeError):
            return

        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            if event.buttons == 1: # Left drag: draw
                self.grid_data[y, x] = 1
            elif event.buttons == 3: # Right drag: erase
                self.grid_data[y, x] = 0
            self.update_canvas()

    def update_canvas(self):
        self.image.set_data(self.grid_data)
        self.fig.canvas.draw_idle()

    def clear_grid(self, event):
        self.grid_data.fill(0)
        self.update_canvas()
        print("캔버스가 초기화되었습니다.")

    def save_hex(self, event):
        print("\n1. im2col 변환을 시작하고 .hex 파일을 생성합니다...")
        generate_im2col_hex(self.grid_data, kernel_size=3, stride=1, filename="im2col_matrix.hex")
        
        print("\n✅ 작업 완료! 생성된 im2col_matrix.hex 파일을 Logisim에서 로드하세요.")
        # HEX 저장 후 그림판 창 닫기
        plt.close(self.fig)
        
    def show(self):
        plt.show()

if __name__ == "__main__":
    print("🎨 마우스로 8x8 이미지를 그리고 .hex 파일을 생성하는 프로그램입니다.")
    print("="*60)
    
    # 대화형 HEX 생성기 시작
    generator = InteractiveHexGenerator()
    generator.show()
    
    print("\n🎉 프로그램 종료") 