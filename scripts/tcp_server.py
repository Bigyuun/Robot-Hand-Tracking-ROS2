import socket
import multiprocessing as mp
import json
import time

class TCPServer:
    def __init__(self, queue_handpose, queue_points):
        self.queue_handpose = queue_handpose
        self.queue_points = queue_points
        self.config = self.load_config('tcp_config.json')
        self.host = self.config['host']
        self.port = self.config['port']
        self.buffer_size = self.config['buffer_size']
        self.manager = mp.Manager()
        self.clients = self.manager.list()  # 공유 리스트
        self.lock = mp.Lock()

        self.pps = 0
        self.count = 0
        self.s_t = time.time()

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            return json.load(file)

    def handle_client(self, client_socket, addr):
        print(f"[*] Handling client {addr[0]}:{addr[1]}")
        try:
            while True:
                data = client_socket.recv(self.buffer_size)
                if not data:
                    break
                print(f"Received data from {addr[0]}:{addr[1]}: {data.decode('utf-8')}")
                self.broadcast(data)
        except Exception as e:
            print(f"Exception in handle_client: {e}")
        finally:
            with self.lock:
                if client_socket in self.clients:
                    self.clients.remove(client_socket)
                    print(f'client({client_socket}) closed.')
            client_socket.close()

    def broadcast(self, data):
        with self.lock:
            for client in self.clients:
                try:
                    client.sendall(data)
                except Exception as e:
                    print(f"Exception in broadcast: {e}")
                    self.clients.remove(client)

    def process_hand_landmarks(self, queue_handpose, queue_points):
        if self.count == 1: # start
            self.s_t = time.time()

        while True:
            try:
                queue_handpose = self.queue_handpose.get()
                queue_points = self.queue_points.get()
            except (mp.queues.Empty):
                print(f'{mp.queues.Empty} queue empty')
                return

            # print(f'========================================================')
            print(f'[TCP Process] pps = {self.pps}')
            # print(f'{queue_handpose}')
            # print(f'========================================================')
            self.pps = self.count / (time.time() - self.s_t)
            self.count += 1

    def start_server(self):
        queue_handpose = mp.Queue(maxsize=1)
        queue_points = mp.Queue(maxsize=1)
        # Hand Landmarks 데이터를 처리할 프로세스를 시작합니다.
        landmarks_process = mp.Process(target=self.process_hand_landmarks, args=(queue_handpose, queue_points))
        landmarks_process.start()

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen(5)  # 최대 5개의 연결을 대기열에 추가할 수 있습니다.
        print(f"[*] Listening on {self.host}:{self.port}")

        try:
            while True:
                client_socket, addr = server.accept()
                print(f"[*] Accepted connection from {addr[0]}:{addr[1]}")

                with self.lock:
                    self.clients.append(client_socket)

                # Start a new process to handle the client
                client_process = mp.Process(target=self.handle_client, args=(client_socket, addr))
                client_process.start()
        except KeyboardInterrupt:
            print("Server shutting down.")
        finally:
            server.close()

            landmarks_process.join()

def start_tcp_server(queue_handpose, queue_points):
    tcp_server = TCPServer(queue_handpose, queue_points)
    tcp_server.start_server()

if __name__ == "__main__":
    queue_handpose = mp.Queue()
    queue_points = mp.Queue()
    plot_process = mp.Process(target=start_tcp_server, args=(queue_handpose, queue_points))
    plot_process.start()
    plot_process.join()
