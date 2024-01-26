import libtorrent as lt
import time

def log_progress(h):
    s = h.status()
    progress = s.progress * 100
    download_rate = s.download_rate / 1000
    upload_rate = s.upload_rate / 1000
    num_peers = s.num_peers
    state = s.state
    print('%.2f%% complete (down: %.1f kB/s up: %.1f kB/s peers: %d) %s' % (
        progress, download_rate, upload_rate, num_peers, state))

def download_torrent(torrent_file, save_path):
    ses = lt.session()

    info = lt.torrent_info(torrent_file)
    h = ses.add_torrent({'ti': info, 'save_path': save_path})

    print('downloading', h.name())
    while not h.is_seed():
        log_progress(h)
        time.sleep(1)  # Add a short delay to avoid spamming the print function
    
    print('download complete')
