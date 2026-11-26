// SmoovPlay Camera: a DirectShow video source ("Video Capture Device" in OBS) fed by SmoovPlay's pipeline.
//
// The pipeline (framegen/vcam.py) writes every frame it shows, with its ideal time, into the shared memory
// "Local\smoovplay-camera" (the layout is in vcam.py's docstring). This filter delivers RGB32 at 60 fps on its own
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

void MakeType(CMediaType* mt, int w, int h) {
    mt->InitMediaType();
    VIDEOINFOHEADER* vih = reinterpret_cast<VIDEOINFOHEADER*>(mt->AllocFormatBuffer(sizeof(VIDEOINFOHEADER)));
    ZeroMemory(vih, sizeof(VIDEOINFOHEADER));
    vih->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    vih->bmiHeader.biWidth = w;
    vih->bmiHeader.biHeight = h;  // positive: rows bottom to top, as RGB DIBs are
    vih->bmiHeader.biPlanes = 1;
    vih->bmiHeader.biBitCount = 32;
    vih->bmiHeader.biCompression = BI_RGB;
    vih->bmiHeader.biSizeImage = (DWORD)w * h * 4;
    vih->AvgTimePerFrame = kFrameTime;
    vih->dwBitRate = (DWORD)((uint64_t)w * h * 32 * kFps);
    mt->SetType(&MEDIATYPE_Video);
    mt->SetSubtype(&MEDIASUBTYPE_RGB32);
    mt->SetFormatType(&FORMAT_VideoInfo);
    mt->SetTemporalCompression(FALSE);
    mt->SetSampleSize(vih->bmiHeader.biSizeImage);
}

bool SizeOf(const AM_MEDIA_TYPE* mt, int* w, int* h) {
    if (!mt || mt->majortype != MEDIATYPE_Video || mt->subtype != MEDIASUBTYPE_RGB32 ||
        mt->formattype != FORMAT_VideoInfo || mt->cbFormat < sizeof(VIDEOINFOHEADER) || !mt->pbFormat)
        return false;
    const VIDEOINFOHEADER* vih = reinterpret_cast<const VIDEOINFOHEADER*>(mt->pbFormat);
    if (vih->bmiHeader.biBitCount != 32 || vih->bmiHeader.biCompression != BI_RGB) return false;
    for (int i = 0; i < kNumSizes; i++)
        if (vih->bmiHeader.biWidth == kSizes[i].w && vih->bmiHeader.biHeight == kSizes[i].h) {
            *w = kSizes[i].w; *h = kSizes[i].h;
            return true;
        }
    return false;
}

// BGR (rows top to bottom, sw x sh) into BGRX (rows bottom to top, dw x dh), scaled nearest-neighbour if the sizes
// differ (OBS asked for another size than SmoovPlay's stream).
void Convert(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh, std::vector<int>* xmap) {
    if (sw == dw && sh == dh) {
        for (int y = 0; y < dh; y++) {
            const uint8_t* s = src + (size_t)y * sw * 3;
            uint32_t* d = reinterpret_cast<uint32_t*>(dst + (size_t)(dh - 1 - y) * dw * 4);
            for (int x = 0; x < dw; x++, s += 3) d[x] = 0xff000000u | (s[2] << 16) | (s[1] << 8) | s[0];
        }
        return;
    }
    if ((int)xmap->size() != dw) {
        xmap->resize(dw);
        for (int x = 0; x < dw; x++) (*xmap)[x] = (int)((int64_t)x * sw / dw) * 3;
    }
    for (int y = 0; y < dh; y++) {
        const uint8_t* row = src + (size_t)((int64_t)y * sh / dh) * sw * 3;
        uint32_t* d = reinterpret_cast<uint32_t*>(dst + (size_t)(dh - 1 - y) * dw * 4);
        for (int x = 0; x < dw; x++) {
            const uint8_t* s = row + (*xmap)[x];
            d[x] = 0xff000000u | (s[2] << 16) | (s[1] << 8) | s[0];
        }
    }
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
        MakeType(&m_mt, w, h);
    }

    DECLARE_IUNKNOWN;

    STDMETHODIMP NonDelegatingQueryInterface(REFIID riid, void** ppv) override {
        if (riid == IID_IAMStreamConfig) return GetInterface(static_cast<IAMStreamConfig*>(this), ppv);
        if (riid == IID_IKsPropertySet) return GetInterface(static_cast<IKsPropertySet*>(this), ppv);
        return CSourceStream::NonDelegatingQueryInterface(riid, ppv);
    }

    // -- media types: RGB32 at 60 fps, in the sizes of kSizes (the current one first)
    HRESULT GetMediaType(int pos, CMediaType* mt) override {
        CAutoLock lock(m_pFilter->pStateLock());
        if (pos < 0) return E_INVALIDARG;
        if (pos == 0) { *mt = m_mt; return S_OK; }
        if (pos > kNumSizes) return VFW_S_NO_MORE_ITEMS;
        MakeType(mt, kSizes[pos - 1].w, kSizes[pos - 1].h);
        return S_OK;
    }

    HRESULT CheckMediaType(const CMediaType* mt) override {
        int w, h;
        return SizeOf(mt, &w, &h) ? S_OK : E_INVALIDARG;
    }

    HRESULT SetMediaType(const CMediaType* mt) override {
        CAutoLock lock(m_pFilter->pStateLock());
        int w, h;
        if (!SizeOf(mt, &w, &h)) return E_INVALIDARG;
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
        {
            CAutoLock lock(m_pFilter->pStateLock());
            const VIDEOINFOHEADER* vih = reinterpret_cast<const VIDEOINFOHEADER*>(m_mt.Format());
            dw = vih->bmiHeader.biWidth;
            dh = vih->bmiHeader.biHeight;
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
        long want = (long)dw * dh * 4;
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
        if (!frame_.empty() && fw_ > 0)
            Convert(frame_.data(), fw_, fh_, data, dw, dh, &xmap_);
        else
            memset(data, 0, want);  // SmoovPlay isn't playing: black

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
        if (!SizeOf(pmt, &w, &h)) return E_INVALIDARG;
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
        *count = kNumSizes;
        *size = sizeof(VIDEO_STREAM_CONFIG_CAPS);
        return S_OK;
    }

    STDMETHODIMP GetStreamCaps(int i, AM_MEDIA_TYPE** ppmt, BYTE* caps) override {
        if (!ppmt || !caps) return E_POINTER;
        if (i < 0 || i >= kNumSizes) return S_FALSE;
        CMediaType mt;
        MakeType(&mt, kSizes[i].w, kSizes[i].h);
        *ppmt = CreateMediaType(&mt);
        if (!*ppmt) return E_OUTOFMEMORY;
        VIDEO_STREAM_CONFIG_CAPS* c = reinterpret_cast<VIDEO_STREAM_CONFIG_CAPS*>(caps);
        ZeroMemory(c, sizeof(*c));
        c->guid = FORMAT_VideoInfo;
        c->InputSize = c->MinCroppingSize = c->MaxCroppingSize = c->MinOutputSize = c->MaxOutputSize =
            SIZE{kSizes[i].w, kSizes[i].h};
        c->CropGranularityX = c->CropGranularityY = c->OutputGranularityX = c->OutputGranularityY = 1;
        c->MinFrameInterval = c->MaxFrameInterval = kFrameTime;
        c->MinBitsPerSecond = c->MaxBitsPerSecond = (LONG)((int64_t)kSizes[i].w * kSizes[i].h * 32 * kFps);
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
    std::vector<int> xmap_;
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
                REGPINTYPES type = {&MEDIATYPE_Video, &MEDIASUBTYPE_RGB32};
                REGFILTERPINS pin = {nullptr, FALSE, TRUE, FALSE, FALSE, &CLSID_NULL, nullptr, 1, &type};
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
