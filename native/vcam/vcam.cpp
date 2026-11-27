// SmoovPlay Camera: a DirectShow video source ("Video Capture Device" in OBS) fed by SmoovPlay's pipeline.
//
// The pipeline (framegen/vcam.py) writes every frame it shows, with its ideal time, into the shared memory
// "Local\smoovplay-camera" (the layout is in vcam.py's docstring). This filter delivers RGB32 (or NV12) at 60 fps on its own
// even beat: on each output tick t it takes the newest frame whose ideal time + D is due (D: the header's delay_ms),
// so the frames go out as evenly as their ideal times, however unevenly SmoovPlay's window showed them. No frame due
// yet: the last one again (counted). SmoovPlay not running: black.
//
// Registration: regsvr32 /s /n /i:user smoovplay-camera.dll registers it for the current user only (no admin;
// an OBS run as administrator doesn't see per-user COM classes), /u with the same arguments removes it.
// DllRegisterServer / DllUnregisterServer do the same for all users (needs admin).
//
// Built on Microsoft's DirectShow base classes (baseclasses/, MIT). MIT licence, as the rest of SmoovPlay.

#include <streams.h>
#include <initguid.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// {FF8D8E89-B9C7-460D-A281-104B34FDCFB0}
DEFINE_GUID(CLSID_SmoovPlayCamera, 0xff8d8e89, 0xb9c7, 0x460d, 0xa2, 0x81, 0x10, 0x4b, 0x34, 0xfd, 0xcf, 0xb0);

namespace {

const wchar_t kName[] = L"SmoovPlay Camera";
const wchar_t kMapName[] = L"Local\\smoovplay-camera";

// The shared memory (framegen/vcam.py)
const uint32_t kMagic = 0x43565053;  // 'SPVC'
const uint32_t kVersion = 1;
const uint32_t kFormatBGR = 1;
const size_t kHeader = 4096, kSlotHeader = 64, kSlots = 8, kMaxW = 2560, kMaxH = 1440;
const size_t kSlotStride = (kSlotHeader + kMaxW * kMaxH * 3 + 4095) / 4096 * 4096;
const size_t kTotal = kHeader + kSlots * kSlotStride;
const uint64_t kWriting = ~0ull;

const int kFps = 60;
const REFERENCE_TIME kFrameTime = 10000000 / kFps;
struct Size { int w, h; };
const Size kSizes[] = {{1920, 1080}, {2560, 1440}, {1280, 720}};
const int kNumSizes = 3;
// Each size as RGB32 (the default: OBS takes it as it is) and NV12 (Discord's video engine takes only YUV formats)
enum Fmt { kRGB32 = 0, kNV12 = 1 };
const int kNumFmts = 2;
const int kNumTypes = kNumSizes * kNumFmts;
DEFINE_GUID(SPVC_MEDIASUBTYPE_NV12, 0x3231564E, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

template <typename T> T Load(const uint8_t* p) { return *reinterpret_cast<const volatile T*>(p); }
template <typename T> void Store(uint8_t* p, T v) { *reinterpret_cast<volatile T*>(p) = v; }

int64_t Qpc() { LARGE_INTEGER v; QueryPerformanceCounter(&v); return v.QuadPart; }
int64_t QpcFreq() { LARGE_INTEGER v; QueryPerformanceFrequency(&v); return v.QuadPart; }

// ---- the shared memory, reader's side
class Ring {
public:
    ~Ring() { Close(); }

    bool Open() {
        if (base_) return true;
        map_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, (DWORD)(kTotal >> 32),
                                  (DWORD)(kTotal & 0xffffffff), kMapName);
        if (!map_) return false;
        base_ = static_cast<uint8_t*>(MapViewOfFile(map_, FILE_MAP_ALL_ACCESS, 0, 0, kTotal));
        if (!base_) { CloseHandle(map_); map_ = nullptr; return false; }
        return true;
    }

    void Close() {
        if (base_) UnmapViewOfFile(base_);
        if (map_) CloseHandle(map_);
        base_ = nullptr; map_ = nullptr;
    }

    // The writer's stream, if it's there and alive: its frame size and D (in QPC units).
    bool Live(int64_t now, int64_t freq, int* w, int* h, int64_t* delay) {
        if (!Open()) return false;
        Store<uint64_t>(base_ + 64, (uint64_t)now);  // reader heartbeat: SmoovPlay copies frames only while it's fresh
        if (Load<uint32_t>(base_) != kMagic || Load<uint32_t>(base_ + 4) != kVersion ||
            Load<uint32_t>(base_ + 24) != kFormatBGR)
            return false;
        uint32_t ww = Load<uint32_t>(base_ + 16), hh = Load<uint32_t>(base_ + 20);
        if (ww == 0 || hh == 0 || ww > kMaxW || hh > kMaxH) return false;
        uint64_t beat = Load<uint64_t>(base_ + 56);
        if (now - (int64_t)beat > freq) return false;  // no frame (or heartbeat) for a second: gone
        *w = (int)ww; *h = (int)hh;
        *delay = (int64_t)Load<uint32_t>(base_ + 32) * freq / 1000;
        return true;
    }

    // The newest frame whose ideal time is at or before `due`, newer than `after`; copied into `out` (w*h*3).
    // Returns its sequence number, or ~0 if there's none.
    uint64_t Take(int64_t due, uint64_t after, int w, int h, std::vector<uint8_t>* out) {
        for (int attempt = 0; attempt < 2; attempt++) {
            uint64_t best = kWriting;
            size_t best_off = 0;
            for (size_t i = 0; i < kSlots; i++) {
                size_t off = kHeader + i * kSlotStride;
                uint64_t seq = Load<uint64_t>(base_ + off);
                if (seq == kWriting || (after != kWriting && seq <= after)) continue;
                int64_t ideal = (int64_t)Load<uint64_t>(base_ + off + 8);
                if (ideal > due) continue;
                if (best == kWriting || seq > best) { best = seq; best_off = off; }
            }
            if (best == kWriting) return kWriting;
            size_t n = (size_t)w * h * 3;
            out->resize(n);
            memcpy(out->data(), base_ + best_off + kSlotHeader, n);
            MemoryBarrier();
            if (Load<uint64_t>(base_ + best_off) == best) return best;  // not overwritten while copying
        }
        return kWriting;
    }

    void Count(uint64_t delivered, uint64_t repeats, uint64_t skipped) {
        if (!base_) return;
        Store<uint64_t>(base_ + 72, delivered);
        Store<uint64_t>(base_ + 80, repeats);
        Store<uint64_t>(base_ + 88, skipped);
    }

private:
    HANDLE map_ = nullptr;
    uint8_t* base_ = nullptr;
};

int Bits(Fmt f) { return f == kNV12 ? 12 : 32; }

void MakeType(CMediaType* mt, int w, int h, Fmt f = kRGB32) {
    mt->InitMediaType();
    VIDEOINFOHEADER* vih = reinterpret_cast<VIDEOINFOHEADER*>(mt->AllocFormatBuffer(sizeof(VIDEOINFOHEADER)));
    ZeroMemory(vih, sizeof(VIDEOINFOHEADER));
    vih->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    vih->bmiHeader.biWidth = w;
    vih->bmiHeader.biHeight = h;  // positive: RGB rows bottom to top, as RGB DIBs are (YUV is top down anyway)
    vih->bmiHeader.biPlanes = 1;
    vih->bmiHeader.biBitCount = (WORD)Bits(f);
    vih->bmiHeader.biCompression = f == kNV12 ? MAKEFOURCC('N', 'V', '1', '2') : BI_RGB;
    vih->bmiHeader.biSizeImage = (DWORD)((int64_t)w * h * Bits(f) / 8);
    vih->AvgTimePerFrame = kFrameTime;
    vih->dwBitRate = (DWORD)((uint64_t)w * h * Bits(f) * kFps);
    mt->SetType(&MEDIATYPE_Video);
    mt->SetSubtype(f == kNV12 ? &SPVC_MEDIASUBTYPE_NV12 : &MEDIASUBTYPE_RGB32);
    mt->SetFormatType(&FORMAT_VideoInfo);
    mt->SetTemporalCompression(FALSE);
    mt->SetSampleSize(vih->bmiHeader.biSizeImage);
}

// One of our types: its size and format
bool TypeOf(const AM_MEDIA_TYPE* mt, int* w, int* h, Fmt* f) {
    if (!mt || mt->majortype != MEDIATYPE_Video || mt->formattype != FORMAT_VideoInfo ||
        mt->cbFormat < sizeof(VIDEOINFOHEADER) || !mt->pbFormat)
        return false;
    const VIDEOINFOHEADER* vih = reinterpret_cast<const VIDEOINFOHEADER*>(mt->pbFormat);
    if (mt->subtype == MEDIASUBTYPE_RGB32 && vih->bmiHeader.biBitCount == 32 &&
        vih->bmiHeader.biCompression == BI_RGB && vih->bmiHeader.biHeight > 0)
        *f = kRGB32;
    else if (mt->subtype == SPVC_MEDIASUBTYPE_NV12 && vih->bmiHeader.biCompression == MAKEFOURCC('N', 'V', '1', '2'))
        *f = kNV12;
    else
        return false;
    int hh = vih->bmiHeader.biHeight < 0 ? -vih->bmiHeader.biHeight : vih->bmiHeader.biHeight;
    for (int i = 0; i < kNumSizes; i++)
        if (vih->bmiHeader.biWidth == kSizes[i].w && hh == kSizes[i].h) {
            *w = kSizes[i].w; *h = kSizes[i].h;
            return true;
        }
    return false;
}

// Source columns for each output column, in bytes into a BGR row: nearest neighbour when OBS or Discord asked for
// another size than SmoovPlay's stream
struct XMap {
    std::vector<int> x;
    int sw = 0, dw = 0;
    const int* For(int s, int d) {
        if (s != sw || d != dw) {
            sw = s; dw = d;
            x.resize(d);
            for (int i = 0; i < d; i++) x[i] = (int)((int64_t)i * s / d) * 3;
        }
        return x.data();
    }
};

// BGR (rows top to bottom, sw x sh) into BGRX (rows bottom to top, dw x dh)
void ToRGB32(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh, XMap* xm) {
    const int* xs = xm->For(sw, dw);
    for (int y = 0; y < dh; y++) {
        const uint8_t* row = src + (size_t)((int64_t)y * sh / dh) * sw * 3;
        uint32_t* d = reinterpret_cast<uint32_t*>(dst + (size_t)(dh - 1 - y) * dw * 4);
        for (int x = 0; x < dw; x++) {
            const uint8_t* p = row + xs[x];
            d[x] = 0xff000000u | (p[2] << 16) | (p[1] << 8) | p[0];
        }
    }
}

// BGR into NV12 (a Y plane, then U and V interleaved at half size), BT.709 limited range as video is. Sizes are
// even (kSizes).
void ToNV12(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh, XMap* xm) {
    const int* xs = xm->For(sw, dw);
    uint8_t* uv = dst + (size_t)dw * dh;
    for (int y = 0; y < dh; y += 2) {
        const uint8_t* r0 = src + (size_t)((int64_t)y * sh / dh) * sw * 3;
        const uint8_t* r1 = src + (size_t)((int64_t)(y + 1) * sh / dh) * sw * 3;
        uint8_t* y0 = dst + (size_t)y * dw;
        uint8_t* y1 = y0 + dw;
        uint8_t* c = uv + (size_t)(y / 2) * dw;
        for (int x = 0; x < dw; x += 2) {
            const uint8_t* p[4] = {r0 + xs[x], r0 + xs[x + 1], r1 + xs[x], r1 + xs[x + 1]};
            uint8_t* out[4] = {y0 + x, y0 + x + 1, y1 + x, y1 + x + 1};
            int sb = 0, sg = 0, sr = 0;
            for (int k = 0; k < 4; k++) {
                int b = p[k][0], g = p[k][1], r = p[k][2];
                *out[k] = (uint8_t)(16 + ((47 * r + 157 * g + 16 * b + 128) >> 8));
                sb += b; sg += g; sr += r;
            }
            c[x] = (uint8_t)(128 + ((-26 * sr - 87 * sg + 112 * sb + 512) >> 10));
            c[x + 1] = (uint8_t)(128 + ((112 * sr - 102 * sg - 10 * sb + 512) >> 10));
        }
    }
}

void Black(uint8_t* dst, int w, int h, Fmt f) {
    if (f == kNV12) {
        memset(dst, 16, (size_t)w * h);
        memset(dst + (size_t)w * h, 128, (size_t)w * h / 2);
    } else {
        memset(dst, 0, (size_t)w * h * 4);
    }
}

// What the camera shows while SmoovPlay isn't sending (not playing, or its camera setting off): a note saying so,
// drawn with GDI as a BGR picture like SmoovPlay's own frames, instead of a black screen that looks broken.
// Empty if GDI couldn't draw it (then black).
std::vector<uint8_t> WaitingPicture(int w, int h) {
    std::vector<uint8_t> out;
    BITMAPINFO bi = {};
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = w;
    bi.bmiHeader.biHeight = -h;  // rows top to bottom
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    HDC dc = CreateCompatibleDC(nullptr);
    if (!dc) return out;
    void* bits = nullptr;
    HBITMAP bmp = CreateDIBSection(dc, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (bmp && bits) {
        HGDIOBJ old = SelectObject(dc, bmp);
        RECT all = {0, 0, w, h};
        HBRUSH bg = CreateSolidBrush(RGB(18, 18, 22));
        FillRect(dc, &all, bg);
        DeleteObject(bg);
        SetBkMode(dc, TRANSPARENT);
        HFONT title_font = CreateFontW(-h / 14, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                                OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, DEFAULT_PITCH,
                                L"Segoe UI");
        HFONT note_font = CreateFontW(-h / 32, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                                  OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, DEFAULT_PITCH,
                                  L"Segoe UI");
        SelectObject(dc, title_font);
        SetTextColor(dc, RGB(255, 176, 46));
        RECT title = {0, h * 2 / 5 - h / 10, w, h * 2 / 5 + h / 20};
        DrawTextW(dc, L"SmoovPlay Camera", -1, &title, DT_CENTER | DT_BOTTOM | DT_SINGLELINE);
        SelectObject(dc, note_font);
        SetTextColor(dc, RGB(225, 225, 230));
        RECT note = {w / 10, h * 2 / 5 + h / 12, w - w / 10, h};
        DrawTextW(dc, L"Waiting for SmoovPlay: the picture appears as soon as it plays.\n"
                      L"If it doesn't, turn on Settings > Streaming > SmoovPlay Camera in SmoovPlay.",
                  -1, &note, DT_CENTER | DT_TOP | DT_WORDBREAK);
        GdiFlush();
        out.resize((size_t)w * h * 3);
        const uint8_t* s = static_cast<const uint8_t*>(bits);
        for (size_t i = 0, n = (size_t)w * h; i < n; i++) {
            out[i * 3] = s[i * 4];
            out[i * 3 + 1] = s[i * 4 + 1];
            out[i * 3 + 2] = s[i * 4 + 2];
        }
        SelectObject(dc, old);
        DeleteObject(title_font);
        DeleteObject(note_font);
    }
    if (bmp) DeleteObject(bmp);
    DeleteDC(dc);
    return out;
}

// ---- the output pin
class CCamPin : public CSourceStream, public IAMStreamConfig, public IKsPropertySet {
public:
    CCamPin(HRESULT* phr, CSource* filter) : CSourceStream(NAME("SmoovPlay Camera pin"), phr, filter, L"Capture") {
        // The stream's own size if SmoovPlay is playing, else 1080p
        int w = 1920, h = 1080, sw, sh;
        int64_t delay;
        if (ring_.Live(Qpc(), QpcFreq(), &sw, &sh, &delay))
            for (int i = 0; i < kNumSizes; i++)
                if (kSizes[i].w == sw && kSizes[i].h == sh) { w = sw; h = sh; }
        MakeType(&m_mt, w, h, kRGB32);
    }

    DECLARE_IUNKNOWN;

    STDMETHODIMP NonDelegatingQueryInterface(REFIID riid, void** ppv) override {
        if (riid == IID_IAMStreamConfig) return GetInterface(static_cast<IAMStreamConfig*>(this), ppv);
        if (riid == IID_IKsPropertySet) return GetInterface(static_cast<IKsPropertySet*>(this), ppv);
        return CSourceStream::NonDelegatingQueryInterface(riid, ppv);
    }

    // -- media types: RGB32 and NV12 at 60 fps, in the sizes of kSizes (the current one first)
    HRESULT GetMediaType(int pos, CMediaType* mt) override {
        CAutoLock lock(m_pFilter->pStateLock());
        if (pos < 0) return E_INVALIDARG;
        if (pos == 0) { *mt = m_mt; return S_OK; }
        if (pos > kNumTypes) return VFW_S_NO_MORE_ITEMS;
        int i = (pos - 1) / kNumFmts;
        MakeType(mt, kSizes[i].w, kSizes[i].h, (Fmt)((pos - 1) % kNumFmts));
        return S_OK;
    }

    HRESULT CheckMediaType(const CMediaType* mt) override {
        int w, h;
        Fmt f;
        return TypeOf(mt, &w, &h, &f) ? S_OK : E_INVALIDARG;
    }

    HRESULT SetMediaType(const CMediaType* mt) override {
        CAutoLock lock(m_pFilter->pStateLock());
        int w, h;
        Fmt f;
        if (!TypeOf(mt, &w, &h, &f)) return E_INVALIDARG;
        return CSourceStream::SetMediaType(mt);
    }

    HRESULT DecideBufferSize(IMemAllocator* alloc, ALLOCATOR_PROPERTIES* req) override {
        CAutoLock lock(m_pFilter->pStateLock());
        const VIDEOINFOHEADER* vih = reinterpret_cast<const VIDEOINFOHEADER*>(m_mt.Format());
        req->cBuffers = max(req->cBuffers, 2L);
        req->cbBuffer = vih->bmiHeader.biSizeImage;
        if (req->cbAlign < 1) req->cbAlign = 1;
        ALLOCATOR_PROPERTIES actual;
        HRESULT hr = alloc->SetProperties(req, &actual);
        if (FAILED(hr)) return hr;
        return actual.cbBuffer < req->cbBuffer ? E_FAIL : S_OK;
    }

    // -- the even beat
    HRESULT OnThreadCreate() override {
        timeBeginPeriod(1);
        timer_ = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        freq_ = QpcFreq();
        started_ = false;
        last_ = kWriting;
        return S_OK;
    }

    HRESULT OnThreadDestroy() override {
        if (timer_) CloseHandle(timer_);
        timer_ = nullptr;
        timeEndPeriod(1);
        return S_OK;
    }

    HRESULT FillBuffer(IMediaSample* sample) override {
        int dw, dh;
        Fmt fmt;
        {
            CAutoLock lock(m_pFilter->pStateLock());
            if (!TypeOf(&m_mt, &dw, &dh, &fmt)) return E_FAIL;
        }
        int64_t now = Qpc();
        if (!started_) { start_ = now; n_ = 0; started_ = true; }
        int64_t target = start_ + n_ * freq_ / kFps;
        if (now - target > 4 * freq_ / kFps) {  // stalled (the graph paused, the PC busy): carry on from now
            n_ = (now - start_) * kFps / freq_;
            target = start_ + n_ * freq_ / kFps;
        }
        WaitUntil(target);

        BYTE* data = nullptr;
        sample->GetPointer(&data);
        long size = sample->GetSize();
        long want = (long)((int64_t)dw * dh * Bits(fmt) / 8);
        if (!data || size < want) return E_FAIL;

        int sw, sh;
        int64_t delay;
        if (ring_.Live(Qpc(), freq_, &sw, &sh, &delay)) {
            uint64_t seq = ring_.Take(target - delay, last_, sw, sh, &frame_);
            if (seq != kWriting) {
                if (last_ != kWriting && seq > last_ + 1) skipped_ += seq - last_ - 1;
                last_ = seq;
                fw_ = sw; fh_ = sh;
                delivered_++;
            } else if (last_ != kWriting) {
                repeats_++;
            }
            ring_.Count(delivered_, repeats_, skipped_);
        } else {
            last_ = kWriting;
            frame_.clear();
        }
        if (frame_.empty() || fw_ <= 0) {  // SmoovPlay isn't sending: the note saying so
            if (waitW_ != dw || waitH_ != dh) {
                waiting_ = WaitingPicture(dw, dh);
                waitW_ = dw; waitH_ = dh;
            }
            if (waiting_.empty())
                Black(data, dw, dh, fmt);
            else if (fmt == kNV12)
                ToNV12(waiting_.data(), dw, dh, data, dw, dh, &xmap_);
            else
                ToRGB32(waiting_.data(), dw, dh, data, dw, dh, &xmap_);
        } else if (fmt == kNV12)
            ToNV12(frame_.data(), fw_, fh_, data, dw, dh, &xmap_);
        else
            ToRGB32(frame_.data(), fw_, fh_, data, dw, dh, &xmap_);

        REFERENCE_TIME t0 = n_ * 10000000LL / kFps, t1 = (n_ + 1) * 10000000LL / kFps;
        sample->SetTime(&t0, &t1);
        sample->SetActualDataLength(want);
        sample->SetSyncPoint(TRUE);
        n_++;
        return S_OK;
    }

    STDMETHODIMP Notify(IBaseFilter*, Quality) override { return E_NOTIMPL; }

    // -- IAMStreamConfig
    STDMETHODIMP SetFormat(AM_MEDIA_TYPE* pmt) override {
        if (!pmt) return E_POINTER;
        int w, h;
        Fmt f;
        if (!TypeOf(pmt, &w, &h, &f)) return E_INVALIDARG;
        CAutoLock lock(m_pFilter->pStateLock());
        if (m_pFilter->IsActive()) return VFW_E_WRONG_STATE;
        m_mt = *pmt;
        IPin* other = GetConnected();
        if (other) {
            if (other->QueryAccept(pmt) != S_OK) return VFW_E_INVALIDMEDIATYPE;
            IFilterGraph* graph = m_pFilter->GetFilterGraph();
            if (graph) graph->Reconnect(this);
        }
        return S_OK;
    }

    STDMETHODIMP GetFormat(AM_MEDIA_TYPE** ppmt) override {
        if (!ppmt) return E_POINTER;
        CAutoLock lock(m_pFilter->pStateLock());
        *ppmt = CreateMediaType(&m_mt);
        return *ppmt ? S_OK : E_OUTOFMEMORY;
    }

    STDMETHODIMP GetNumberOfCapabilities(int* count, int* size) override {
        if (!count || !size) return E_POINTER;
        *count = kNumTypes;
        *size = sizeof(VIDEO_STREAM_CONFIG_CAPS);
        return S_OK;
    }

    STDMETHODIMP GetStreamCaps(int i, AM_MEDIA_TYPE** ppmt, BYTE* caps) override {
        if (!ppmt || !caps) return E_POINTER;
        if (i < 0 || i >= kNumTypes) return S_FALSE;
        Fmt f = (Fmt)(i % kNumFmts);
        i /= kNumFmts;
        CMediaType mt;
        MakeType(&mt, kSizes[i].w, kSizes[i].h, f);
        *ppmt = CreateMediaType(&mt);
        if (!*ppmt) return E_OUTOFMEMORY;
        VIDEO_STREAM_CONFIG_CAPS* c = reinterpret_cast<VIDEO_STREAM_CONFIG_CAPS*>(caps);
        ZeroMemory(c, sizeof(*c));
        c->guid = FORMAT_VideoInfo;
        c->InputSize = c->MinCroppingSize = c->MaxCroppingSize = c->MinOutputSize = c->MaxOutputSize =
            SIZE{kSizes[i].w, kSizes[i].h};
        c->CropGranularityX = c->CropGranularityY = c->OutputGranularityX = c->OutputGranularityY = 1;
        c->MinFrameInterval = c->MaxFrameInterval = kFrameTime;
        c->MinBitsPerSecond = c->MaxBitsPerSecond = (LONG)((int64_t)kSizes[i].w * kSizes[i].h * Bits(f) * kFps);
        return S_OK;
    }

    // -- IKsPropertySet: a capture pin
    STDMETHODIMP Set(REFGUID, DWORD, LPVOID, DWORD, LPVOID, DWORD) override { return E_NOTIMPL; }

    STDMETHODIMP Get(REFGUID set, DWORD id, LPVOID, DWORD, LPVOID data, DWORD size, DWORD* returned) override {
        if (set != AMPROPSETID_Pin) return E_PROP_SET_UNSUPPORTED;
        if (id != AMPROPERTY_PIN_CATEGORY) return E_PROP_ID_UNSUPPORTED;
        if (!data && !returned) return E_POINTER;
        if (returned) *returned = sizeof(GUID);
        if (!data) return S_OK;
        if (size < sizeof(GUID)) return E_UNEXPECTED;
        *reinterpret_cast<GUID*>(data) = PIN_CATEGORY_CAPTURE;
        return S_OK;
    }

    STDMETHODIMP QuerySupported(REFGUID set, DWORD id, DWORD* support) override {
        if (set != AMPROPSETID_Pin) return E_PROP_SET_UNSUPPORTED;
        if (id != AMPROPERTY_PIN_CATEGORY) return E_PROP_ID_UNSUPPORTED;
        if (support) *support = KSPROPERTY_SUPPORT_GET;
        return S_OK;
    }

private:
    void WaitUntil(int64_t target) {
        int64_t left = target - Qpc();
        if (left <= 0) return;
        if (timer_ && left > freq_ / 1000) {  // the high-resolution timer for most of it...
            LARGE_INTEGER due;
            due.QuadPart = -(left - freq_ / 2000) * 10000000LL / freq_;  // 100 ns units, relative
            if (SetWaitableTimer(timer_, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(timer_, 100);
        }
        while (Qpc() < target) YieldProcessor();  // ...and a short spin for the rest
    }

    Ring ring_;
    HANDLE timer_ = nullptr;
    int64_t freq_ = 1, start_ = 0, n_ = 0;
    bool started_ = false;
    uint64_t last_ = kWriting, delivered_ = 0, repeats_ = 0, skipped_ = 0;
    std::vector<uint8_t> frame_;
    XMap xmap_;
    std::vector<uint8_t> waiting_;  // WaitingPicture at the output's size
    int waitW_ = 0, waitH_ = 0;
    int fw_ = 0, fh_ = 0;
};

// ---- the filter
class CCamFilter : public CSource {
public:
    static CUnknown* WINAPI CreateInstance(LPUNKNOWN unk, HRESULT* phr) {
        CCamFilter* f = new CCamFilter(unk, phr);
        if (!f && phr) *phr = E_OUTOFMEMORY;
        return f;
    }

    STDMETHODIMP NonDelegatingQueryInterface(REFIID riid, void** ppv) override {
        // Some apps ask the filter itself for the pin's interfaces
        if ((riid == IID_IAMStreamConfig || riid == IID_IKsPropertySet) && m_iPins > 0 && m_paStreams[0])
            return m_paStreams[0]->QueryInterface(riid, ppv);
        return CSource::NonDelegatingQueryInterface(riid, ppv);
    }

private:
    CCamFilter(LPUNKNOWN unk, HRESULT* phr) : CSource(NAME("SmoovPlay Camera"), unk, CLSID_SmoovPlayCamera) {
        new CCamPin(phr, this);  // adds itself to the filter's pins
    }
};

// ---- registration
HRESULT Register(bool on) {
    wchar_t path[MAX_PATH];
    if (!GetModuleFileNameW(g_hInst, path, MAX_PATH)) return HRESULT_FROM_WIN32(GetLastError());
    wchar_t clsid[64];
    StringFromGUID2(CLSID_SmoovPlayCamera, clsid, 64);
    std::wstring key = std::wstring(L"CLSID\\") + clsid;

    HRESULT hr = CoInitialize(nullptr);
    bool uninit = SUCCEEDED(hr);
    IFilterMapper2* fm = nullptr;
    hr = CoCreateInstance(CLSID_FilterMapper2, nullptr, CLSCTX_INPROC_SERVER, IID_IFilterMapper2,
                          reinterpret_cast<void**>(&fm));
    if (SUCCEEDED(hr)) {
        if (on) {
            LSTATUS r = RegSetKeyValueW(HKEY_CLASSES_ROOT, key.c_str(), nullptr, REG_SZ, kName, sizeof(kName));
            std::wstring server = key + L"\\InprocServer32";
            if (r == ERROR_SUCCESS)
                r = RegSetKeyValueW(HKEY_CLASSES_ROOT, server.c_str(), nullptr, REG_SZ, path,
                                    (DWORD)((wcslen(path) + 1) * sizeof(wchar_t)));
            if (r == ERROR_SUCCESS)
                r = RegSetKeyValueW(HKEY_CLASSES_ROOT, server.c_str(), L"ThreadingModel", REG_SZ, L"Both",
                                    sizeof(L"Both"));
            hr = HRESULT_FROM_WIN32(r);
            if (SUCCEEDED(hr)) {
                REGPINTYPES types[] = {{&MEDIATYPE_Video, &MEDIASUBTYPE_RGB32},
                                       {&MEDIATYPE_Video, &SPVC_MEDIASUBTYPE_NV12}};
                REGFILTERPINS pin = {nullptr, FALSE, TRUE, FALSE, FALSE, &CLSID_NULL, nullptr, 2, types};
                REGFILTER2 rf2;
                rf2.dwVersion = 1;
                rf2.dwMerit = MERIT_DO_NOT_USE;
                rf2.cPins = 1;
                rf2.rgPins = &pin;
                hr = fm->RegisterFilter(CLSID_SmoovPlayCamera, kName, nullptr, &CLSID_VideoInputDeviceCategory,
                                        kName, &rf2);
            }
        } else {
            fm->UnregisterFilter(&CLSID_VideoInputDeviceCategory, kName, CLSID_SmoovPlayCamera);
            LSTATUS r = RegDeleteTreeW(HKEY_CLASSES_ROOT, key.c_str());
            hr = (r == ERROR_SUCCESS || r == ERROR_FILE_NOT_FOUND) ? S_OK : HRESULT_FROM_WIN32(r);
        }
        fm->Release();
    }
    if (uninit) CoUninitialize();
    return hr;
}

}  // namespace

CFactoryTemplate g_Templates[] = {
    {kName, &CLSID_SmoovPlayCamera, CCamFilter::CreateInstance, nullptr, nullptr},
};
int g_cTemplates = sizeof(g_Templates) / sizeof(g_Templates[0]);

STDAPI DllRegisterServer() { return Register(true); }
STDAPI DllUnregisterServer() { return Register(false); }

// regsvr32 /n /i:user: for the current user only (HKCU\Software\Classes), no admin
STDAPI DllInstall(BOOL install, LPCWSTR cmd) {
    bool user = cmd && _wcsicmp(cmd, L"user") == 0;
    HKEY classes = nullptr;
    if (user) {
        if (RegCreateKeyExW(HKEY_CURRENT_USER, L"Software\\Classes", 0, nullptr, 0, KEY_ALL_ACCESS, nullptr,
                            &classes, nullptr) != ERROR_SUCCESS)
            return E_FAIL;
        RegOverridePredefKey(HKEY_CLASSES_ROOT, classes);
    }
    HRESULT hr = Register(install != FALSE);
    if (user) {
        RegOverridePredefKey(HKEY_CLASSES_ROOT, nullptr);
        RegCloseKey(classes);
    }
    return hr;
}

extern "C" BOOL WINAPI DllEntryPoint(HINSTANCE, ULONG, LPVOID);

BOOL APIENTRY DllMain(HANDLE module, DWORD reason, LPVOID reserved) {
    return DllEntryPoint(reinterpret_cast<HINSTANCE>(module), reason, reserved);
}
