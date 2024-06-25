import multiprocessing as mp
from handlandmarks_show_plot import start_real_time_plot
from handlandmarks_with_realsense import handlandmarks_with_realsense

if __name__ == '__main__':
    # queue_handpose = mp.Manager().Queue(maxsize=1)
    # queue_points = mp.Manager().Queue(maxsize=1)
    queue_handpose = mp.Queue(maxsize=1)
    queue_points = mp.Queue(maxsize=1)

    data_process = mp.Process(target=handlandmarks_with_realsense, args=(queue_handpose, queue_points))
    plot_process = mp.Process(target=start_real_time_plot, args=(queue_handpose, queue_points))

    data_process.start()
    plot_process.start()

    data_process.join()
    plot_process.join()
