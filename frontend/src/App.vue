<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import axios from 'axios'
import Lightbox from './components/Lightbox.vue'
import LandingPage from './components/LandingPage.vue'
import Login from './components/Login.vue'
import History from './components/History.vue'
import Toast from './components/Toast.vue'

// ================= 类型定义 =================
interface Detection {
  class_cn: string
  confidence: number
  bbox: number[]
  dimensions?: string
}

interface Model {
  name: string
  path: string
}

interface BatchResult {
  filename: string
  detections: Detection[]
  image_base64: string
}

// ================= 状态管理 =================
const showLanding = ref(true)
const showLogin = ref(false)
const currentUser = ref<string | null>(localStorage.getItem('username'))
const currentTab = ref<'detect' | 'compare' | 'history'>('detect')
const toastRef = ref<InstanceType<typeof Toast> | null>(null)

// Lightbox 状态
const lightboxImage = ref<string | null>(null)
const openLightbox = (src: string | null) => {
  if (src) lightboxImage.value = src
}

// 实时检测状态
const selectedFiles = ref<FileList | null>(null)
const previewUrl = ref<string | null>(null)
const resultImage = ref<string | null>(null)
const detections = ref<Detection[]>([])
const loading = ref(false)
const stats = ref({ time: 0, count: 0, model: '' })

// 批量模式
const isBatchMode = ref(false)
const batchResults = ref<BatchResult[]>([])
const currentBatchIndex = ref(0)
const MAX_BATCH_SIZE = 16

// 模型列表
const availableModels = ref<Model[]>([])
const selectedModel = ref<string>('yolo11n (Official)')

// 对比模式状态
const compareFile = ref<File | null>(null)
const comparePreviewUrl = ref<string | null>(null)
const compareModelA = ref<string>('')
const compareModelB = ref<string>('')
const compareResultA = ref<string | null>(null)
const compareResultB = ref<string | null>(null)
const compareStatsA = ref({ time: 0, count: 0 })
const compareStatsB = ref({ time: 0, count: 0 })
const compareDetectionsA = ref<Detection[]>([])
const compareDetectionsB = ref<Detection[]>([])
const compareLoading = ref(false)

const confFilter = ref(0.25)
const filteredDetections = computed(() => {
  return detections.value.filter(d => d.confidence >= confFilter.value)
})

// ================= 逻辑函数 =================
const showToast = (msg: string, type: 'success' | 'error' | 'info' = 'success') => {
  toastRef.value?.showToast(msg, type)
}

const handleTabChange = (tab: 'detect' | 'compare' | 'history') => {
  if (tab === 'history' && !currentUser.value) {
    showLogin.value = true
    showToast('请先登录以查看历史记录', 'info')
    return
  }
  currentTab.value = tab
}

const handleLoginSuccess = (username: string) => {
  currentUser.value = username
  showLogin.value = false
  currentTab.value = 'history'
  showToast(`欢迎回来，${username}！`, 'success')
}

const logout = () => {
  localStorage.removeItem('username')
  localStorage.removeItem('token')
  currentUser.value = null
  if (currentTab.value === 'history') {
    currentTab.value = 'detect'
  }
  showToast('已安全退出登录', 'info')
}

const fetchModels = async () => {
  try {
    const res = await axios.get('http://127.0.0.1:8000/models')
    availableModels.value = res.data
    if (availableModels.value.length > 0) {
      compareModelA.value = availableModels.value[0].name
      compareModelB.value = availableModels.value.length > 1 ? availableModels.value[1].name : availableModels.value[0].name
    }
  } catch (e) {
    console.error('获取模型列表失败', e)
    showToast('无法连接到后端服务', 'error')
  }
}

const handleModelChange = async () => {
  const model = availableModels.value.find(m => m.name === selectedModel.value)
  if (model) {
    try {
      await axios.post('http://127.0.0.1:8000/set_model', model)
      showToast(`已切换至模型: ${model.name}`, 'success')
    } catch (e) {
      showToast('模型切换失败', 'error')
    }
  }
}

const onFileChange = (e: Event) => {
  const target = e.target as HTMLInputElement
  if (target.files && target.files.length > 0) {
    if (target.files.length > MAX_BATCH_SIZE) {
      showToast(`一次最多只能上传 ${MAX_BATCH_SIZE} 张图片，已自动截取`, 'info')
      const dt = new DataTransfer()
      for (let i = 0; i < MAX_BATCH_SIZE; i++) {
        dt.items.add(target.files[i])
      }
      selectedFiles.value = dt.files
    } else {
      selectedFiles.value = target.files
    }

    isBatchMode.value = selectedFiles.value!.length > 1

    const file = selectedFiles.value![0]
    previewUrl.value = URL.createObjectURL(file)
    resultImage.value = null
    detections.value = []
    batchResults.value = []
    currentBatchIndex.value = 0
  }
}

const onCompareFileChange = (e: Event) => {
  const target = e.target as HTMLInputElement
  if (target.files && target.files.length > 0) {
    compareFile.value = target.files[0]
    comparePreviewUrl.value = URL.createObjectURL(compareFile.value)
    compareResultA.value = null
    compareResultB.value = null
    compareDetectionsA.value = []
    compareDetectionsB.value = []
  }
}

const uploadAndDetect = async () => {
  if (!selectedFiles.value) return
  loading.value = true

  const formData = new FormData()
  if (isBatchMode.value) {
    for (let i = 0; i < selectedFiles.value.length; i++) {
      formData.append('files', selectedFiles.value[i])
    }
  } else {
    formData.append('file', selectedFiles.value[0])
  }

  formData.append('conf', '0.25')
  formData.append('iou', '0.45')
  if (currentUser.value) formData.append('username', currentUser.value)

  try {
    const endpoint = isBatchMode.value ? 'http://127.0.0.1:8000/detect/batch' : 'http://127.0.0.1:8000/detect'
    const response = await axios.post(endpoint, formData)
    const data = response.data

    if (data.success) {
      if (isBatchMode.value) {
        batchResults.value = data.results
        showBatchResult(0)
        stats.value = {
          time: data.total_inference_time_ms,
          count: batchResults.value.reduce((acc, cur) => acc + cur.detections.length, 0),
          model: data.model_used
        }
        showToast(`批量检测完成，共 ${selectedFiles.value.length} 张`, 'success')
      } else {
        resultImage.value = data.image_base64
        detections.value = data.detections
        stats.value = {
          time: data.inference_time_ms,
          count: data.detections.length,
          model: data.model_used
        }
        showToast(`检测完成，发现 ${data.detections.length} 处缺陷`, 'success')
      }
    }
  } catch (error) {
    showToast('检测失败，请检查后端服务', 'error')
  } finally {
    loading.value = false
  }
}

const showBatchResult = (index: number) => {
  if (index >= 0 && index < batchResults.value.length) {
    currentBatchIndex.value = index
    const res = batchResults.value[index]
    resultImage.value = res.image_base64
    detections.value = res.detections
  }
}

const runComparison = async () => {
  if (!compareFile.value) return
  compareLoading.value = true

  try {
    // 1. 跑模型 A
    const modelA = availableModels.value.find(m => m.name === compareModelA.value)
    if (modelA) await axios.post('http://127.0.0.1:8000/set_model', modelA)

    const formDataA = new FormData()
    formDataA.append('file', compareFile.value)
    formDataA.append('conf', '0.25')
    const resA = await axios.post('http://127.0.0.1:8000/detect', formDataA)

    if (resA.data.success) {
      compareResultA.value = resA.data.image_base64
      compareDetectionsA.value = resA.data.detections
      compareStatsA.value = {
        time: resA.data.inference_time_ms,
        count: resA.data.detections.length
      }
    }

    // 2. 跑模型 B
    const modelB = availableModels.value.find(m => m.name === compareModelB.value)
    if (modelB) await axios.post('http://127.0.0.1:8000/set_model', modelB)

    const formDataB = new FormData()
    formDataB.append('file', compareFile.value)
    formDataB.append('conf', '0.25')
    const resB = await axios.post('http://127.0.0.1:8000/detect', formDataB)

    if (resB.data.success) {
      compareResultB.value = resB.data.image_base64
      compareDetectionsB.value = resB.data.detections
      compareStatsB.value = {
        time: resB.data.inference_time_ms,
        count: resB.data.detections.length
      }
    }

    const originalModel = availableModels.value.find(m => m.name === selectedModel.value)
    if (originalModel) await axios.post('http://127.0.0.1:8000/set_model', originalModel)

    showToast('对比分析完成', 'success')

  } catch (e) {
    showToast('对比分析失败', 'error')
  } finally {
    compareLoading.value = false
  }
}

const submitFeedback = async (type: 'false_positive' | 'false_negative', details: string) => {
  if (!selectedFiles.value) return
  const filename = isBatchMode.value ? batchResults.value[currentBatchIndex.value].filename : selectedFiles.value[0].name

  try {
    await axios.post('http://127.0.0.1:8000/feedback', {
      filename: filename,
      model_name: stats.value.model,
      feedback_type: type,
      details: details
    })
    showToast('感谢您的反馈！', 'success')
  } catch (e) {
    showToast('反馈提交失败', 'error')
  }
}

onMounted(fetchModels)
</script>

<template>
  <!-- 全局 Toast -->
  <Toast ref="toastRef" />

  <!-- Landing Page -->
  <Transition name="slide-up">
    <LandingPage v-if="showLanding" @start="showLanding = false" />
  </Transition>

  <!-- Login Modal -->
  <Login v-if="showLogin" @login-success="handleLoginSuccess" @close="showLogin = false" />

  <div class="min-h-screen bg-[#f8fafc] text-slate-900 font-sans selection:bg-blue-100">
    <!-- 全局 Lightbox 组件 -->
    <Lightbox :src="lightboxImage" @close="lightboxImage = null" />

    <!-- Top Navigation -->
    <nav class="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200 px-6 py-3">
      <div class="max-w-7xl mx-auto flex justify-between items-center">
        <div class="flex items-center gap-3 cursor-pointer" @click="showLanding = true">
          <div class="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-200">
            <span class="text-white text-xl font-bold">⚡</span>
          </div>
          <div>
            <h1 class="text-lg font-bold tracking-tight">PowerAI <span class="text-blue-600">Inspection</span></h1>
            <p class="text-[10px] text-slate-400 font-medium uppercase tracking-widest">Intelligent Detection System</p>
          </div>
        </div>

        <div class="flex bg-slate-100 p-1 rounded-lg">
          <button
            @click="handleTabChange('detect')"
            :class="['px-4 py-1.5 text-xs font-bold rounded-md transition-all', currentTab === 'detect' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700']"
          >
            实时检测
          </button>
          <button
            @click="handleTabChange('compare')"
            :class="['px-4 py-1.5 text-xs font-bold rounded-md transition-all', currentTab === 'compare' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700']"
          >
            模型对比
          </button>
          <button
            @click="handleTabChange('history')"
            :class="['px-4 py-1.5 text-xs font-bold rounded-md transition-all', currentTab === 'history' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700']"
          >
            历史记录
          </button>
        </div>

        <div class="flex items-center gap-4">
          <!-- 用户信息 -->
          <div v-if="currentUser" class="flex items-center gap-2 pl-4 border-l border-slate-200">
            <span class="text-sm font-bold text-slate-700">{{ currentUser }}</span>
            <button @click="logout" class="text-xs text-red-500 hover:underline">退出</button>
          </div>
          <div v-else class="flex items-center gap-2 pl-4 border-l border-slate-200">
            <button @click="showLogin = true" class="text-sm font-bold text-blue-600 hover:underline">登录</button>
          </div>
        </div>
      </div>
    </nav>

    <main class="max-w-7xl mx-auto p-6 lg:p-8">

      <!-- ==================== 实时检测视图 ==================== -->
      <div v-if="currentTab === 'detect'" class="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <!-- Left Column: Controls & Stats -->
        <div class="lg:col-span-4 space-y-6">
          <div class="bg-white rounded-3xl p-8 shadow-sm border border-slate-200 transition-all hover:shadow-xl hover:shadow-slate-200/50">
            <h3 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">模型选择</h3>
            <select
              v-model="selectedModel"
              @change="handleModelChange"
              class="w-full text-sm font-semibold bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 focus:ring-2 focus:ring-blue-500 outline-none mb-6"
            >
              <option v-for="m in availableModels" :key="m.name" :value="m.name">{{ m.name }}</option>
            </select>

            <h3 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-6">数据输入</h3>
            <div
              class="relative group border-2 border-dashed border-slate-200 rounded-2xl p-8 transition-all hover:border-blue-400 hover:bg-blue-50/50 text-center cursor-pointer"
              @click="$refs.fileInput.click()"
            >
              <input type="file" ref="fileInput" class="hidden" @change="onFileChange" multiple accept="image/*" />
              <div class="flex flex-col items-center gap-3">
                <div class="w-12 h-12 bg-slate-50 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                  <span class="text-2xl">📸</span>
                </div>
                <div>
                  <p class="text-sm font-bold text-slate-700">点击上传巡检图</p>
                  <p class="text-xs text-slate-400 mt-1">支持批量上传 (多选, Max 16)</p>
                </div>
              </div>
            </div>
            <div v-if="selectedFiles && selectedFiles.length > 0" class="mt-4 text-center">
              <span class="text-xs font-bold bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
                已选择 {{ selectedFiles.length }} 张图片
              </span>
            </div>
            <button
              @click="uploadAndDetect"
              :disabled="!selectedFiles || loading"
              class="w-full mt-6 py-4 bg-slate-900 hover:bg-blue-600 disabled:bg-slate-200 text-white font-bold rounded-2xl shadow-lg shadow-slate-200 transition-all flex justify-center items-center gap-3 group"
            >
              <span v-if="loading" class="animate-spin text-xl">⏳</span>
              <span v-else class="group-hover:translate-x-1 transition-transform">🚀</span>
              {{ loading ? 'AI 分析中...' : '开始智能识别' }}
            </button>
          </div>

          <div v-if="isBatchMode && batchResults.length > 0" class="bg-white rounded-3xl p-6 shadow-sm border border-slate-200">
            <h3 class="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">批量结果浏览</h3>
            <div class="flex items-center justify-between gap-4">
              <button @click="showBatchResult(currentBatchIndex - 1)" :disabled="currentBatchIndex === 0" class="p-2 rounded-full hover:bg-slate-100 disabled:opacity-30">⬅️</button>
              <span class="text-sm font-bold text-slate-700">{{ currentBatchIndex + 1 }} / {{ batchResults.length }}</span>
              <button @click="showBatchResult(currentBatchIndex + 1)" :disabled="currentBatchIndex === batchResults.length - 1" class="p-2 rounded-full hover:bg-slate-100 disabled:opacity-30">➡️</button>
            </div>
            <p class="text-center text-xs text-slate-400 mt-2 truncate">{{ batchResults[currentBatchIndex].filename }}</p>
          </div>

          <div v-if="stats.time > 0" class="bg-blue-600 rounded-3xl p-8 text-white shadow-lg shadow-blue-200 relative overflow-hidden">
            <div class="relative z-10">
              <h3 class="text-xs font-bold text-blue-200 uppercase tracking-widest mb-6">分析报告</h3>
              <div class="grid grid-cols-2 gap-4">
                <div class="bg-white/10 backdrop-blur-md rounded-2xl p-4">
                  <p class="text-[10px] text-blue-100 uppercase font-bold">总耗时</p>
                  <p class="text-2xl font-mono font-bold">{{ stats.time }}<span class="text-xs ml-1">ms</span></p>
                </div>
                <div class="bg-white/10 backdrop-blur-md rounded-2xl p-4">
                  <p class="text-[10px] text-blue-100 uppercase font-bold">发现缺陷</p>
                  <p class="text-2xl font-mono font-bold">{{ stats.count }}<span class="text-xs ml-1">处</span></p>
                </div>
              </div>
              <p class="mt-6 text-[10px] text-blue-200 italic">使用模型: {{ stats.model }}</p>

              <button
                @click="submitFeedback('false_negative', '用户标记漏检')"
                class="mt-4 w-full py-2 bg-white/20 hover:bg-white/30 text-xs font-bold rounded-lg transition-colors"
              >
                ⚠️ 标记为漏检 (False Negative)
              </button>
            </div>
            <div class="absolute -right-4 -bottom-4 w-24 h-24 bg-white/10 rounded-full blur-2xl"></div>
          </div>
        </div>

        <!-- Right Column: Visualization -->
        <div class="lg:col-span-8 space-y-6">
          <div class="bg-white rounded-3xl shadow-sm border border-slate-200 overflow-hidden">
            <div class="flex border-b border-slate-100">
              <button class="px-6 py-4 text-sm font-bold border-b-2 border-blue-600 text-blue-600">视觉分析</button>
            </div>
            <div class="p-6">
              <!-- 使用自定义 Lightbox -->
              <div
                class="aspect-video bg-slate-900 rounded-2xl border border-slate-800 flex items-center justify-center overflow-hidden relative group cursor-zoom-in"
                @click="openLightbox(resultImage)"
              >
                <img v-if="resultImage" :src="resultImage" class="h-full object-contain" />
                <div v-else class="text-slate-700 flex flex-col items-center gap-2">
                  <span class="text-4xl opacity-20">🔍</span>
                </div>
                <div v-if="loading" class="absolute inset-0 bg-slate-900/80 backdrop-blur-sm flex items-center justify-center pointer-events-none">
                  <div class="flex flex-col items-center gap-4">
                    <div class="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                    <p class="text-blue-400 font-mono text-xs tracking-widest animate-pulse">PROCESSING...</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="bg-white rounded-3xl shadow-sm border border-slate-200 overflow-hidden">
            <div class="px-8 py-6 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
              <div>
                <h3 class="font-bold text-slate-800">缺陷识别清单</h3>
                <p class="text-xs text-slate-400 mt-0.5">基于当前置信度阈值过滤</p>
              </div>
              <div class="flex items-center gap-4">
                <span class="text-[10px] font-bold text-slate-400 uppercase">过滤阈值</span>
                <input type="range" v-model="confFilter" min="0" max="1" step="0.01" class="w-24 accent-blue-600" />
                <span class="text-xs font-mono font-bold text-blue-600 w-8">{{ (confFilter * 100).toFixed(0) }}%</span>
              </div>
            </div>
            <div class="overflow-x-auto">
              <table class="w-full text-left border-collapse">
                <thead>
                  <tr>
                    <th class="px-8 py-4 text-[10px] font-bold text-slate-400 uppercase tracking-widest">缺陷类型</th>
                    <th class="px-8 py-4 text-[10px] font-bold text-slate-400 uppercase tracking-widest">可信度分数</th>
                    <th class="px-8 py-4 text-[10px] font-bold text-slate-400 uppercase tracking-widest">目标尺寸 (WxH)</th>
                    <th class="px-8 py-4 text-[10px] font-bold text-slate-400 uppercase tracking-widest">操作</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-slate-50">
                  <tr v-for="(item, index) in filteredDetections" :key="index" class="group hover:bg-blue-50/30 transition-colors">
                    <td class="px-8 py-5">
                      <div class="flex items-center gap-3">
                        <div class="w-2 h-2 bg-blue-500 rounded-full shadow-[0_0_8px_rgba(59,130,246,0.5)]"></div>
                        <span class="text-sm font-bold text-slate-700">{{ item.class_cn }}</span>
                      </div>
                    </td>
                    <td class="px-8 py-5">
                      <div class="flex items-center gap-3">
                        <div class="flex-1 h-1 bg-slate-100 rounded-full max-w-[80px] overflow-hidden">
                          <div class="h-full bg-blue-600 transition-all duration-1000" :style="{ width: (item.confidence * 100) + '%' }"></div>
                        </div>
                        <span class="text-xs font-mono font-bold text-slate-500">{{ (item.confidence * 100).toFixed(1) }}%</span>
                      </div>
                    </td>
                    <td class="px-8 py-5">
                      <span class="text-xs font-mono font-bold text-slate-600 bg-slate-100 px-2 py-1 rounded">{{ item.dimensions || '-' }}</span>
                    </td>
                    <td class="px-8 py-5">
                      <button
                        @click="submitFeedback('false_positive', `误检: ${item.class_cn}`)"
                        class="text-[10px] font-bold text-red-500 hover:bg-red-50 px-2 py-1 rounded transition-colors"
                      >
                        ❌ 误检
                      </button>
                    </td>
                  </tr>
                  <tr v-if="filteredDetections.length === 0">
                    <td colspan="4" class="px-8 py-20 text-center">
                      <div class="flex flex-col items-center gap-3 opacity-20">
                        <span class="text-5xl">📡</span>
                        <p class="text-sm font-bold uppercase tracking-widest">No Anomalies Detected</p>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <!-- ==================== 模型对比视图 ==================== -->
      <div v-else-if="currentTab === 'compare'" class="space-y-8">
        <div class="bg-white rounded-3xl p-8 shadow-sm border border-slate-200">
          <div class="flex flex-col md:flex-row justify-between items-center gap-6">
            <div class="flex-1 w-full">
              <label class="block text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">模型 A (基准)</label>
              <select v-model="compareModelA" class="w-full text-sm font-semibold bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 focus:ring-2 focus:ring-blue-500 outline-none">
                <option v-for="m in availableModels" :key="m.name" :value="m.name">{{ m.name }}</option>
              </select>
            </div>
            <div class="text-2xl text-slate-300 font-black">VS</div>
            <div class="flex-1 w-full">
              <label class="block text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">模型 B (对照)</label>
              <select v-model="compareModelB" class="w-full text-sm font-semibold bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 focus:ring-2 focus:ring-blue-500 outline-none">
                <option v-for="m in availableModels" :key="m.name" :value="m.name">{{ m.name }}</option>
              </select>
            </div>
            <button
              @click="runComparison"
              :disabled="!compareFile || compareLoading"
              class="px-8 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-200 text-white font-bold rounded-xl shadow-lg shadow-blue-200 transition-all flex items-center gap-2"
            >
              <span v-if="compareLoading" class="animate-spin">⏳</span>
              开始对比
            </button>
          </div>

          <div class="mt-6">
            <div
              class="relative group border-2 border-dashed border-slate-200 rounded-2xl p-6 transition-all hover:border-blue-400 hover:bg-blue-50/50 text-center cursor-pointer"
              @click="$refs.compareInput.click()"
            >
              <input type="file" ref="compareInput" class="hidden" @change="onCompareFileChange" accept="image/*" />
              <div v-if="!compareFile" class="flex flex-col items-center gap-2">
                <span class="text-2xl">📸</span>
                <p class="text-sm font-bold text-slate-400">点击上传对比图片</p>
              </div>
              <div v-else class="flex items-center justify-center gap-4">
                <img :src="comparePreviewUrl" class="h-16 w-16 object-cover rounded-lg border border-slate-200" />
                <div class="text-left">
                  <p class="text-sm font-bold text-slate-700">已选择: {{ compareFile.name }}</p>
                  <p class="text-xs text-blue-500 font-bold">点击更换</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div v-if="compareResultA || compareResultB" class="grid grid-cols-1 md:grid-cols-2 gap-8">
          <!-- Result A -->
          <div class="space-y-4">
            <div class="flex justify-between items-center px-2">
              <span class="font-bold text-slate-700">模型 A 结果</span>
              <span class="text-xs font-mono bg-blue-100 text-blue-700 px-2 py-1 rounded">{{ compareStatsA.time }}ms</span>
            </div>
            <div
              class="aspect-video bg-slate-900 rounded-2xl border border-slate-800 flex items-center justify-center overflow-hidden cursor-zoom-in"
              @click="openLightbox(compareResultA)"
            >
              <img v-if="compareResultA" :src="compareResultA" class="h-full object-contain" />
              <div v-else class="text-slate-600">Waiting...</div>
            </div>

            <!-- 缺陷清单 A -->
            <div class="bg-white rounded-2xl border border-slate-200 p-4">
              <h4 class="text-xs font-bold text-slate-400 uppercase mb-3">缺陷识别清单 ({{ compareStatsA.count }})</h4>
              <div class="space-y-2 max-h-40 overflow-y-auto">
                <div v-for="(det, idx) in compareDetectionsA" :key="idx" class="flex justify-between text-xs p-2 bg-slate-50 rounded">
                  <span class="font-bold text-slate-700">{{ det.class_cn }}</span>
                  <span class="font-mono text-blue-600">{{ (det.confidence * 100).toFixed(0) }}%</span>
                </div>
                <div v-if="compareDetectionsA.length === 0" class="text-center text-xs text-slate-400 italic">无检测结果</div>
              </div>
            </div>
          </div>

          <!-- Result B -->
          <div class="space-y-4">
            <div class="flex justify-between items-center px-2">
              <span class="font-bold text-slate-700">模型 B 结果</span>
              <span class="text-xs font-mono bg-green-100 text-green-700 px-2 py-1 rounded">{{ compareStatsB.time }}ms</span>
            </div>
            <div
              class="aspect-video bg-slate-900 rounded-2xl border border-slate-800 flex items-center justify-center overflow-hidden cursor-zoom-in"
              @click="openLightbox(compareResultB)"
            >
              <img v-if="compareResultB" :src="compareResultB" class="h-full object-contain" />
              <div v-else class="text-slate-600">Waiting...</div>
            </div>

            <!-- 缺陷清单 B -->
            <div class="bg-white rounded-2xl border border-slate-200 p-4">
              <h4 class="text-xs font-bold text-slate-400 uppercase mb-3">缺陷识别清单 ({{ compareStatsB.count }})</h4>
              <div class="space-y-2 max-h-40 overflow-y-auto">
                <div v-for="(det, idx) in compareDetectionsB" :key="idx" class="flex justify-between text-xs p-2 bg-slate-50 rounded">
                  <span class="font-bold text-slate-700">{{ det.class_cn }}</span>
                  <span class="font-mono text-green-600">{{ (det.confidence * 100).toFixed(0) }}%</span>
                </div>
                <div v-if="compareDetectionsB.length === 0" class="text-center text-xs text-slate-400 italic">无检测结果</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ==================== 历史记录视图 ==================== -->
      <div v-else-if="currentTab === 'history'">
        <History :username="currentUser" />
      </div>

    </main>
  </div>
</template>

<style scoped>
::-webkit-scrollbar {
  width: 6px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: #e2e8f0;
  border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
  background: #cbd5e1;
}

select {
  appearance: none;
  background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2364748b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");
  background-repeat: no-repeat;
  background-position: right 0.75rem center;
  background-size: 1em;
  padding-right: 2.5rem;
}

.slide-up-enter-active,
.slide-up-leave-active {
  transition: transform 0.5s ease-in-out;
}

.slide-up-enter-from,
.slide-up-leave-to {
  transform: translateY(-100%);
}
</style>
