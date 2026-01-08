<script setup lang="ts">
import { ref, onMounted, computed, onUnmounted } from 'vue'
import axios from 'axios'

interface Detection {
  class_cn: string
  confidence: number
  bbox: number[]
}

interface Model {
  name: string
  path: string
}

interface HistoryItem {
  id: string
  timestamp: string
  originalImage: string
  resultImage: string
  model: string
  count: number
  time: number
  detections: Detection[]
}

// --- Navigation State ---
const currentView = ref<'detect' | 'history'>('detect')

// --- Detection State ---
const selectedFile = ref<File | null>(null)
const previewUrl = ref<string | null>(null)
const resultImage = ref<string | null>(null)
const detections = ref<Detection[]>([])
const loading = ref(false)
const stats = ref({ time: 0, count: 0, model: '' })

const availableModels = ref<Model[]>([])
const selectedModel = ref<string>('yolo11n (Official)')

const confFilter = ref(0.25)
const filteredDetections = computed(() => {
  return detections.value.filter(d => d.confidence >= confFilter.value)
})

// --- History Logic ---
const history = ref<HistoryItem[]>([])
const itemsPerPage = 7
const currentPage = ref(1)

const totalPages = computed(() => Math.ceil(history.value.length / itemsPerPage))
const paginatedHistory = computed(() => {
  const start = (currentPage.value - 1) * itemsPerPage
  const end = start + itemsPerPage
  return history.value.slice(start, end)
})

const loadHistory = () => {
  const saved = localStorage.getItem('power_vision_history')
  if (saved) history.value = JSON.parse(saved)
}

const saveHistory = () => {
  try {
    localStorage.setItem('power_vision_history', JSON.stringify(history.value))
  } catch (e) {
    console.warn('Storage quota exceeded, removing oldest item...')
    if (history.value.length > 1) {
      history.value.pop()
      saveHistory()
    }
  }
}

const addToHistory = (item: Omit<HistoryItem, 'id' | 'timestamp'>) => {
  const newItem: HistoryItem = {
    ...item,
    id: Date.now().toString(),
    timestamp: new Date().toLocaleString()
  }
  history.value.unshift(newItem)
  if (history.value.length > 50) history.value.pop() // 压缩后可以存更多，设为50条
  saveHistory()
}

const clearHistory = () => {
  if (confirm('确定要清空所有历史记录吗？')) {
    history.value = []
    currentPage.value = 1
    saveHistory()
  }
}

const viewHistoryItem = (item: HistoryItem) => {
  previewUrl.value = item.originalImage
  resultImage.value = item.resultImage
  detections.value = item.detections
  stats.value = {
    time: item.time,
    count: item.count,
    model: item.model
  }
  selectedModel.value = item.model
  currentView.value = 'detect'
}

// --- Utils: Image Compression ---
const compressImage = (base64: string, maxWidth = 800): Promise<string> => {
  return new Promise((resolve) => {
    const img = new Image()
    img.src = base64
    img.onload = () => {
      const canvas = document.createElement('canvas')
      let width = img.width
      let height = img.height
      if (width > maxWidth) {
        height = (maxWidth / width) * height
        width = maxWidth
      }
      canvas.width = width
      canvas.height = height
      const ctx = canvas.getContext('2d')
      ctx?.drawImage(img, 0, 0, width, height)
      resolve(canvas.toDataURL('image/jpeg', 0.7))
    }
  })
}

const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = error => reject(error)
  })
}

// --- Image Preview & Zoom Logic ---
const isModalOpen = ref(false)
const modalImageUrl = ref('')
const zoomLevel = ref(1)

const openModal = (url: string) => {
  modalImageUrl.value = url
  isModalOpen.value = true
  zoomLevel.value = 1
  document.body.style.overflow = 'hidden'
}

const closeModal = () => {
  isModalOpen.value = false
  document.body.style.overflow = ''
}

const handleZoom = (delta: number) => {
  const newZoom = zoomLevel.value + delta
  if (newZoom >= 0.5 && newZoom <= 5) {
    zoomLevel.value = newZoom
  }
}

const onWheel = (e: WheelEvent) => {
  if (isModalOpen.value) {
    e.preventDefault()
    handleZoom(e.deltaY > 0 ? -0.1 : 0.1)
  }
}

onMounted(() => {
  fetchModels()
  loadHistory()
  window.addEventListener('wheel', onWheel, { passive: false })
})

onUnmounted(() => {
  window.removeEventListener('wheel', onWheel)
})

const fetchModels = async () => {
  try {
    const res = await axios.get('http://127.0.0.1:8000/models')
    availableModels.value = res.data
  } catch (e) {
    console.error('获取模型列表失败', e)
  }
}

const handleModelChange = async () => {
  const model = availableModels.value.find(m => m.name === selectedModel.value)
  if (model) {
    try {
      await axios.post('http://127.0.0.1:8000/set_model', model)
    } catch (e) {
      alert('模型切换失败')
    }
  }
}

const onFileChange = (e: Event) => {
  const target = e.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    selectedFile.value = file
    previewUrl.value = URL.createObjectURL(file)
    resultImage.value = null
    detections.value = []
    stats.value = { time: 0, count: 0, model: '' }
  }
}

const uploadAndDetect = async () => {
  if (!selectedFile.value) return
  loading.value = true
  const formData = new FormData()
  formData.append('file', selectedFile.value)
  formData.append('conf', confFilter.value.toString())

  try {
    const originalBase64 = await fileToBase64(selectedFile.value)
    const response = await axios.post('http://127.0.0.1:8000/detect', formData)
    const data = response.data
    if (data.success) {
      resultImage.value = data.image_base64
      detections.value = data.detections
      stats.value = {
        time: data.inference_time_ms,
        count: data.detections.length,
        model: data.model_used
      }
      const compressedOriginal = await compressImage(originalBase64)
      const compressedResult = await compressImage(data.image_base64)
      addToHistory({
        originalImage: compressedOriginal,
        resultImage: compressedResult,
        model: data.model_used,
        count: data.detections.length,
        time: data.inference_time_ms,
        detections: data.detections
      })
    }
  } catch (error: any) {
    console.error('Detection Error:', error)
    alert('检测失败，请检查后端服务')
  } finally {
    loading.value = false
  }
}

const resetDetect = () => {
  selectedFile.value = null
  previewUrl.value = null
  resultImage.value = null
  detections.value = []
  stats.value = { time: 0, count: 0, model: '' }
}
</script>

<template>
  <div class="flex min-h-screen bg-[#F9FAFB] text-[#111827] font-sans">
    <!-- Sidebar -->
    <aside class="w-[260px] bg-[#1A1A1A] text-[#9CA3AF] flex flex-col fixed h-full z-50">
      <div class="p-6 flex items-center gap-3 border-b border-gray-800">
        <div class="w-8 h-8 bg-[#3B82F6] rounded-lg flex items-center justify-center shadow-lg shadow-blue-900/20">
          <span class="text-white text-sm font-bold">AI</span>
        </div>
        <h1 class="text-white font-bold tracking-tight text-lg">PowerVision</h1>
      </div>

      <nav class="flex-1 p-4 space-y-1">
        <div class="px-3 py-2 text-[10px] font-bold uppercase tracking-widest text-gray-500 mb-2">Main Menu</div>
        <button
          @click="currentView = 'detect'"
          :class="[currentView === 'detect' ? 'bg-gray-800 text-white' : 'hover:bg-gray-800 hover:text-white']"
          class="w-full flex items-center gap-3 px-3 py-2.5 rounded-md transition-colors text-left"
        >
          <span class="text-lg">📊</span>
          <span class="text-sm font-medium">智能检测</span>
        </button>
        <button
          @click="currentView = 'history'"
          :class="[currentView === 'history' ? 'bg-gray-800 text-white' : 'hover:bg-gray-800 hover:text-white']"
          class="w-full flex items-center gap-3 px-3 py-2.5 rounded-md transition-colors text-left"
        >
          <span class="text-lg">📁</span>
          <span class="text-sm font-medium">历史记录</span>
        </button>
      </nav>

      <div class="p-4 border-t border-gray-800">
        <div class="flex items-center gap-3 px-3 py-2">
          <div class="w-8 h-8 bg-gray-700 rounded-full flex items-center justify-center text-xs text-white">JD</div>
          <div class="flex-1 min-w-0">
            <p class="text-xs font-bold text-white truncate">Admin User</p>
            <p class="text-[10px] truncate">admin@power.ai</p>
          </div>
        </div>
      </div>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 ml-[260px]">
      <!-- Header -->
      <header class="h-16 bg-white border-b border-[#E5E7EB] flex items-center justify-between px-8 sticky top-0 z-40">
        <div class="flex items-center gap-2 text-sm">
          <span class="text-gray-400">项目</span>
          <span class="text-gray-300">/</span>
          <span class="font-medium">{{ currentView === 'detect' ? '智能检测' : '历史记录' }}</span>
        </div>

        <div class="flex items-center gap-4">
          <div class="flex items-center gap-2 px-3 py-1 bg-green-50 rounded-full border border-green-100">
            <div class="w-1.5 h-1.5 bg-green-500 rounded-full"></div>
            <span class="text-[10px] font-bold text-green-600 uppercase">Service Online</span>
          </div>
          <button
            v-if="currentView === 'detect' && resultImage"
            @click="resetDetect"
            class="text-xs font-bold text-gray-500 hover:text-blue-600 px-3 py-1.5 border border-gray-200 rounded-md hover:bg-gray-50 transition-colors"
          >
            重新上传
          </button>
          <button
            v-if="currentView === 'history'"
            @click="clearHistory"
            class="text-xs font-bold text-red-500 hover:text-red-600 px-3 py-1.5 border border-red-100 rounded-md hover:bg-red-50 transition-colors"
          >
            清空记录
          </button>
        </div>
      </header>

      <!-- View: Detection -->
      <div v-if="currentView === 'detect'" class="p-8 min-h-[calc(100vh-64px)] flex flex-col items-center">
        <Transition name="fade-scale" mode="out-in">
          <div v-if="!resultImage && !loading" class="w-full max-w-2xl mt-12">
            <div class="bg-white rounded-xl border border-[#E5E7EB] shadow-sm p-8">
              <div class="flex justify-between items-center mb-8">
                <div>
                  <h2 class="text-xl font-bold text-gray-800">新建检测任务</h2>
                  <p class="text-xs text-gray-400 mt-1">上传巡检图像并配置 AI 模型参数</p>
                </div>
                <select
                  v-model="selectedModel"
                  @change="handleModelChange"
                  class="text-xs font-semibold bg-gray-50 border border-[#E5E7EB] rounded-md px-4 py-2 focus:ring-1 focus:ring-blue-500 outline-none cursor-pointer"
                >
                  <option v-for="m in availableModels" :key="m.name" :value="m.name">{{ m.name }}</option>
                </select>
              </div>

              <div
                class="relative aspect-video border-2 border-dashed border-[#E5E7EB] rounded-xl overflow-hidden group cursor-pointer hover:border-blue-400 hover:bg-blue-50/30 transition-all"
                @click="$refs.fileInput.click()"
              >
                <input type="file" ref="fileInput" class="hidden" @change="onFileChange" />
                <div v-if="previewUrl" class="w-full h-full relative">
                  <img :src="previewUrl" class="w-full h-full object-contain" />
                  <div class="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-4">
                    <button @click.stop="openModal(previewUrl!)" class="p-2 bg-white/20 hover:bg-white/40 rounded-full text-white backdrop-blur-md">🔍 预览</button>
                    <button @click.stop="selectedFile = null; previewUrl = null" class="p-2 bg-white/20 hover:bg-red-500/40 rounded-full text-white backdrop-blur-md">✕ 移除</button>
                  </div>
                </div>
                <div v-else class="w-full h-full flex flex-col items-center justify-center gap-4">
                  <div class="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center text-3xl group-hover:scale-110 transition-transform">📤</div>
                  <div class="text-center">
                    <p class="text-sm font-bold text-gray-600">点击或拖拽上传图像</p>
                    <p class="text-[10px] text-gray-400 mt-1 uppercase tracking-widest">支持 JPG, PNG, BMP (Max 10MB)</p>
                  </div>
                </div>
              </div>

              <div class="mt-8 p-6 bg-gray-50 rounded-xl">
                <div class="flex justify-between items-center mb-4">
                  <span class="text-xs font-bold text-gray-500 uppercase tracking-wider">置信度阈值 (Confidence)</span>
                  <span class="text-sm font-mono font-bold text-blue-600 bg-white px-2 py-1 rounded border border-blue-100">{{ (confFilter * 100).toFixed(0) }}%</span>
                </div>
                <input type="range" v-model="confFilter" min="0" max="1" step="0.01" class="w-full accent-[#3B82F6]" />
              </div>

              <button
                @click="uploadAndDetect"
                :disabled="!selectedFile || loading"
                class="w-full mt-8 py-4 bg-[#3B82F6] hover:bg-blue-700 disabled:bg-gray-200 text-white text-sm font-bold rounded-xl shadow-lg shadow-blue-200 transition-all flex justify-center items-center gap-3"
              >
                🚀 开始智能识别
              </button>
            </div>
          </div>

          <div v-else-if="loading" class="w-full max-w-2xl mt-32 flex flex-col items-center gap-8">
            <div class="relative">
              <div class="w-24 h-24 border-4 border-blue-100 border-t-blue-500 rounded-full animate-spin"></div>
              <div class="absolute inset-0 flex items-center justify-center text-2xl">🧠</div>
            </div>
            <div class="text-center">
              <h2 class="text-xl font-bold text-gray-800 animate-pulse">AI 正在深度分析中...</h2>
              <p class="text-xs text-gray-400 mt-2 uppercase tracking-widest">正在识别电力设备缺陷并生成报告</p>
            </div>
          </div>

          <div v-else-if="resultImage" class="w-full max-w-6xl">
            <div class="grid grid-cols-12 gap-8">
              <div class="col-span-4 space-y-6">
                <div class="bg-white rounded-xl border border-[#E5E7EB] p-6">
                  <h3 class="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-4">当前配置</h3>
                  <div class="space-y-3">
                    <div class="flex justify-between text-xs">
                      <span class="text-gray-500">模型</span>
                      <span class="font-bold">{{ selectedModel }}</span>
                    </div>
                    <div class="flex justify-between text-xs">
                      <span class="text-gray-500">阈值</span>
                      <span class="font-bold text-blue-600">{{ (confFilter * 100).toFixed(0) }}%</span>
                    </div>
                  </div>
                </div>

                <Transition name="slide-up" appear>
                  <div class="bg-white rounded-xl border border-[#E5E7EB] p-6 shadow-sm">
                    <h3 class="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-6">推理统计报告</h3>
                    <div class="space-y-6">
                      <div class="flex justify-between items-end">
                        <div>
                          <p class="text-[10px] text-gray-400 uppercase font-bold">Inference Time</p>
                          <p class="text-3xl font-mono font-bold text-[#111827]">{{ stats.time }}<span class="text-sm ml-1">ms</span></p>
                        </div>
                        <div class="w-24 h-1 bg-gray-100 rounded-full overflow-hidden mb-2">
                          <div class="bg-blue-500 h-full" :style="{ width: Math.min(stats.time/10, 100) + '%' }"></div>
                        </div>
                      </div>
                      <div class="pt-4 border-t border-gray-50">
                        <p class="text-[10px] text-gray-400 uppercase font-bold mb-1">Detected Objects</p>
                        <p class="text-3xl font-mono font-bold text-blue-600">{{ stats.count }}</p>
                      </div>
                    </div>
                  </div>
                </Transition>
              </div>

              <div class="col-span-8 space-y-6">
                <Transition name="stagger" appear>
                  <div class="bg-white rounded-xl border border-[#E5E7EB] overflow-hidden shadow-sm">
                    <div class="px-6 py-4 border-b border-[#E5E7EB] flex justify-between items-center bg-gray-50/50">
                      <h3 class="text-sm font-bold">视觉分析结果</h3>
                    </div>
                    <div class="p-6">
                      <div class="grid grid-cols-2 gap-6">
                        <div class="space-y-2">
                          <p class="text-[10px] font-bold text-gray-400 uppercase">Original Source</p>
                          <div class="aspect-square bg-gray-50 rounded-lg border border-[#E5E7EB] overflow-hidden cursor-zoom-in group relative" @click="openModal(previewUrl!)">
                            <img :src="previewUrl!" class="w-full h-full object-contain transition-transform group-hover:scale-105" />
                          </div>
                        </div>
                        <div class="space-y-2">
                          <p class="text-[10px] font-bold text-blue-500 uppercase">AI Inference</p>
                          <div class="aspect-square bg-[#1A1A1A] rounded-lg border border-gray-800 overflow-hidden cursor-zoom-in group relative" @click="openModal(resultImage!)">
                            <img v-if="resultImage" :src="resultImage" class="w-full h-full object-contain transition-transform group-hover:scale-105" />
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </Transition>

                <Transition name="stagger" appear>
                  <div class="bg-white rounded-xl border border-[#E5E7EB] overflow-hidden shadow-sm">
                    <table class="w-full text-left border-collapse">
                      <thead>
                        <tr class="bg-gray-50 border-b border-[#E5E7EB]">
                          <th class="px-6 py-3 text-[10px] font-bold text-gray-500 uppercase tracking-wider">缺陷类型</th>
                          <th class="px-6 py-3 text-[10px] font-bold text-gray-500 uppercase tracking-wider">置信度</th>
                          <th class="px-6 py-3 text-[10px] font-bold text-gray-500 uppercase tracking-wider">状态</th>
                        </tr>
                      </thead>
                      <tbody class="divide-y divide-[#E5E7EB]">
                        <tr v-for="(item, index) in filteredDetections" :key="index" class="hover:bg-gray-50 transition-colors">
                          <td class="px-6 py-4 text-sm font-medium text-gray-700">{{ item.class_cn }}</td>
                          <td class="px-6 py-4">
                            <div class="flex items-center gap-3">
                              <div class="flex-1 h-1.5 bg-gray-100 rounded-full max-w-[100px]">
                                <div class="h-full bg-[#3B82F6] rounded-full" :style="{ width: (item.confidence * 100) + '%' }"></div>
                              </div>
                              <span class="text-xs font-mono text-gray-500">{{ (item.confidence * 100).toFixed(1) }}%</span>
                            </div>
                          </td>
                          <td class="px-6 py-4">
                            <span class="px-2 py-0.5 bg-blue-50 text-[#3B82F6] text-[10px] font-bold rounded border border-blue-100">已检出</span>
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </Transition>
              </div>
            </div>
          </div>
        </Transition>
      </div>

      <!-- View: History -->
      <div v-else class="p-8 max-w-6xl mx-auto">
        <div class="bg-white rounded-xl border border-[#E5E7EB] overflow-hidden shadow-sm flex flex-col min-h-[600px]">
          <div class="flex-1">
            <table class="w-full text-left border-collapse">
              <thead>
                <tr class="bg-gray-50 border-b border-[#E5E7EB]">
                  <th class="px-6 py-4 text-[10px] font-bold text-gray-500 uppercase tracking-wider">预览</th>
                  <th class="px-6 py-4 text-[10px] font-bold text-gray-500 uppercase tracking-wider">检测时间</th>
                  <th class="px-6 py-4 text-[10px] font-bold text-gray-500 uppercase tracking-wider">使用模型</th>
                  <th class="px-6 py-4 text-[10px] font-bold text-gray-500 uppercase tracking-wider">结果统计</th>
                  <th class="px-6 py-4 text-[10px] font-bold text-gray-500 uppercase tracking-wider">操作</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-[#E5E7EB]">
                <tr v-for="item in paginatedHistory" :key="item.id" class="hover:bg-gray-50 transition-colors group">
                  <td class="px-6 py-4">
                    <div class="w-12 h-12 bg-gray-900 rounded border border-gray-800 overflow-hidden cursor-pointer" @click="openModal(item.resultImage)">
                      <img :src="item.resultImage" class="w-full h-full object-cover" />
                    </div>
                  </td>
                  <td class="px-6 py-4">
                    <p class="text-sm font-medium text-gray-700">{{ item.timestamp }}</p>
                  </td>
                  <td class="px-6 py-4">
                    <span class="px-2 py-1 bg-gray-100 text-gray-600 text-[10px] font-bold rounded">{{ item.model }}</span>
                  </td>
                  <td class="px-6 py-4">
                    <div class="flex flex-col gap-1">
                      <span class="text-xs text-gray-600 font-medium">检出: <span class="text-blue-600 font-bold">{{ item.count }}</span></span>
                      <span class="text-[10px] text-gray-400 font-mono">耗时: {{ item.time }}ms</span>
                    </div>
                  </td>
                  <td class="px-6 py-4">
                    <button @click="viewHistoryItem(item)" class="text-xs font-bold text-[#3B82F6] hover:underline">查看详情</button>
                  </td>
                </tr>
                <tr v-if="history.length === 0">
                  <td colspan="5" class="px-6 py-24 text-center text-gray-400 text-xs font-bold uppercase tracking-widest">暂无历史记录</td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- Pagination Controls -->
          <div v-if="totalPages > 1" class="px-6 py-4 border-t border-[#E5E7EB] flex items-center justify-between bg-gray-50/30">
            <div class="text-xs text-gray-500">
              显示第 {{ (currentPage - 1) * itemsPerPage + 1 }} 到 {{ Math.min(currentPage * itemsPerPage, history.length) }} 条，共 {{ history.length }} 条
            </div>
            <div class="flex items-center gap-2">
              <button
                @click="currentPage--"
                :disabled="currentPage === 1"
                class="px-3 py-1.5 text-xs font-bold border border-[#E5E7EB] rounded-md bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                上一页
              </button>
              <div class="flex items-center gap-1">
                <button
                  v-for="p in totalPages" :key="p"
                  @click="currentPage = p"
                  :class="[currentPage === p ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-600 border-[#E5E7EB] hover:bg-gray-50']"
                  class="w-8 h-8 text-xs font-bold border rounded-md transition-all"
                >
                  {{ p }}
                </button>
              </div>
              <button
                @click="currentPage++"
                :disabled="currentPage === totalPages"
                class="px-3 py-1.5 text-xs font-bold border border-[#E5E7EB] rounded-md bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                下一页
              </button>
            </div>
          </div>
        </div>
      </div>
    </main>

    <!-- Image Modal -->
    <Transition name="fade">
      <div v-if="isModalOpen" class="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 backdrop-blur-sm" @click.self="closeModal">
        <div class="absolute top-6 right-6 flex items-center gap-4 z-[110]">
          <div class="flex items-center bg-white/10 rounded-full p-1 border border-white/20">
            <button @click="handleZoom(-0.2)" class="w-8 h-8 flex items-center justify-center text-white hover:bg-white/20 rounded-full transition-colors">➖</button>
            <span class="px-3 text-xs font-mono text-white min-w-[60px] text-center">{{ (zoomLevel * 100).toFixed(0) }}%</span>
            <button @click="handleZoom(0.2)" class="w-8 h-8 flex items-center justify-center text-white hover:bg-white/20 rounded-full transition-colors">➕</button>
          </div>
          <button @click="closeModal" class="w-10 h-10 flex items-center justify-center bg-white/10 hover:bg-red-500 text-white rounded-full transition-all border border-white/20">✕</button>
        </div>
        <div class="w-full h-full flex items-center justify-center overflow-auto p-12" @click.self="closeModal">
          <img :src="modalImageUrl" class="max-w-none transition-transform duration-200 ease-out shadow-2xl" :style="{ transform: `scale(${zoomLevel})` }" />
        </div>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
/* 极简滚动条 */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #E5E7EB; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #D1D5DB; }

select {
  appearance: none;
  background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%239CA3AF' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");
  background-repeat: no-repeat;
  background-position: right 0.75rem center;
  background-size: 0.8em;
  padding-right: 2.5rem;
}

/* Transitions */
.fade-enter-active, .fade-leave-active { transition: opacity 0.3s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

.fade-scale-enter-active, .fade-scale-leave-active { transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1); }
.fade-scale-enter-from { opacity: 0; transform: scale(0.95) translateY(10px); }
.fade-scale-leave-to { opacity: 0; transform: scale(1.05) translateY(-10px); }

.slide-up-enter-active { transition: all 0.5s ease-out; transition-delay: 0.2s; }
.slide-up-enter-from { opacity: 0; transform: translateY(30px); }

.stagger-enter-active { transition: all 0.6s ease-out; transition-delay: 0.4s; }
.stagger-enter-from { opacity: 0; transform: translateX(40px); }
</style>
