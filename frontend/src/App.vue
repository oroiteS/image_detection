<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
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

const selectedFile = ref<File | null>(null)
const previewUrl = ref<string | null>(null)
const resultImage = ref<string | null>(null)
const detections = ref<Detection[]>([])
const loading = ref(false)
const stats = ref({ time: 0, count: 0, model: '' })

const availableModels = ref<Model[]>([])
const selectedModel = ref<string>('yolo11n (Official)')

// 过滤逻辑：前端实时过滤（可选）
const confFilter = ref(0.25)
const filteredDetections = computed(() => {
  return detections.value.filter(d => d.confidence >= confFilter.value)
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
  }
}

const uploadAndDetect = async () => {
  if (!selectedFile.value) return
  loading.value = true
  const formData = new FormData()
  formData.append('file', selectedFile.value)

  try {
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
    }
  } catch (error) {
    alert('检测失败，请检查后端服务')
  } finally {
    loading.value = false
  }
}

onMounted(fetchModels)
</script>

<template>
  <div class="min-h-screen bg-[#f8fafc] text-slate-900 font-sans selection:bg-blue-100">
    <!-- Top Navigation -->
    <nav class="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200 px-6 py-3">
      <div class="max-w-7xl mx-auto flex justify-between items-center">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-200">
            <span class="text-white text-xl font-bold">⚡</span>
          </div>
          <div>
            <h1 class="text-lg font-bold tracking-tight">PowerAI <span class="text-blue-600">Inspection</span></h1>
            <p class="text-[10px] text-slate-400 font-medium uppercase tracking-widest">Intelligent Detection System</p>
          </div>
        </div>

        <div class="flex items-center gap-4">
          <div class="hidden md:flex items-center gap-2 px-3 py-1.5 bg-slate-100 rounded-full border border-slate-200">
            <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span class="text-xs font-bold text-slate-600">后端已连接</span>
          </div>
          <select
            v-model="selectedModel"
            @change="handleModelChange"
            class="text-sm font-semibold bg-white border border-slate-200 rounded-lg px-3 py-1.5 focus:ring-2 focus:ring-blue-500 outline-none cursor-pointer shadow-sm"
          >
            <option v-for="m in availableModels" :key="m.name" :value="m.name">{{ m.name }}</option>
          </select>
        </div>
      </div>
    </nav>

    <main class="max-w-7xl mx-auto p-6 lg:p-8">
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">

        <!-- Left Column: Controls & Stats -->
        <div class="lg:col-span-4 space-y-6">
          <!-- Upload Card -->
          <div class="bg-white rounded-3xl p-8 shadow-sm border border-slate-200 transition-all hover:shadow-xl hover:shadow-slate-200/50">
            <h3 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-6">数据输入</h3>

            <div
              class="relative group border-2 border-dashed border-slate-200 rounded-2xl p-8 transition-all hover:border-blue-400 hover:bg-blue-50/50 text-center cursor-pointer"
              @click="$refs.fileInput.click()"
            >
              <input type="file" ref="fileInput" class="hidden" @change="onFileChange" />
              <div class="flex flex-col items-center gap-3">
                <div class="w-12 h-12 bg-slate-50 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                  <span class="text-2xl">📸</span>
                </div>
                <div>
                  <p class="text-sm font-bold text-slate-700">点击上传巡检图</p>
                  <p class="text-xs text-slate-400 mt-1">支持 JPG, PNG, BMP 格式</p>
                </div>
              </div>
            </div>

            <button
              @click="uploadAndDetect"
              :disabled="!selectedFile || loading"
              class="w-full mt-6 py-4 bg-slate-900 hover:bg-blue-600 disabled:bg-slate-200 text-white font-bold rounded-2xl shadow-lg shadow-slate-200 transition-all flex justify-center items-center gap-3 group"
            >
              <span v-if="loading" class="animate-spin text-xl">⏳</span>
              <span v-else class="group-hover:translate-x-1 transition-transform">🚀</span>
              {{ loading ? 'AI 分析中...' : '开始智能识别' }}
            </button>
          </div>

          <!-- Stats Card -->
          <div v-if="stats.time > 0" class="bg-blue-600 rounded-3xl p-8 text-white shadow-lg shadow-blue-200 relative overflow-hidden">
            <div class="relative z-10">
              <h3 class="text-xs font-bold text-blue-200 uppercase tracking-widest mb-6">分析报告</h3>
              <div class="grid grid-cols-2 gap-4">
                <div class="bg-white/10 backdrop-blur-md rounded-2xl p-4">
                  <p class="text-[10px] text-blue-100 uppercase font-bold">推理耗us</p>
                  <p class="text-2xl font-mono font-bold">{{ stats.time }}<span class="text-xs ml-1">ms</span></p>
                </div>
                <div class="bg-white/10 backdrop-blur-md rounded-2xl p-4">
                  <p class="text-[10px] text-blue-100 uppercase font-bold">发现缺陷</p>
                  <p class="text-2xl font-mono font-bold">{{ stats.count }}<span class="text-xs ml-1">处</span></p>
                </div>
              </div>
              <p class="mt-6 text-[10px] text-blue-200 italic">使用模型: {{ stats.model }}</p>
            </div>
            <!-- Decorative circles -->
            <div class="absolute -right-4 -bottom-4 w-24 h-24 bg-white/10 rounded-full blur-2xl"></div>
          </div>
        </div>

        <!-- Right Column: Visualization -->
        <div class="lg:col-span-8 space-y-6">
          <!-- Image Viewer -->
          <div class="bg-white rounded-3xl shadow-sm border border-slate-200 overflow-hidden">
            <div class="flex border-b border-slate-100">
              <button class="px-6 py-4 text-sm font-bold border-b-2 border-blue-600 text-blue-600">视觉分析</button>
              <button class="px-6 py-4 text-sm font-bold text-slate-400 hover:text-slate-600 transition-colors">原始数据</button>
            </div>

            <div class="p-6">
              <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="space-y-3">
                  <span class="text-[10px] font-black text-slate-300 uppercase tracking-tighter">Input Source</span>
                  <div class="aspect-square bg-slate-50 rounded-2xl border border-slate-100 flex items-center justify-center overflow-hidden group relative">
                    <img v-if="previewUrl" :src="previewUrl" class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105" />
                    <div v-else class="text-slate-300 flex flex-col items-center gap-2">
                      <span class="text-4xl opacity-20">🖼️</span>
                    </div>
                  </div>
                </div>
                <div class="space-y-3">
                  <span class="text-[10px] font-black text-blue-300 uppercase tracking-tighter">AI Inference</span>
                  <div class="aspect-square bg-slate-900 rounded-2xl border border-slate-800 flex items-center justify-center overflow-hidden relative group">
                    <img v-if="resultImage" :src="resultImage" class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105" />
                    <div v-else class="text-slate-700 flex flex-col items-center gap-2">
                      <span class="text-4xl opacity-20">🔍</span>
                    </div>
                    <!-- Loading Overlay -->
                    <div v-if="loading" class="absolute inset-0 bg-slate-900/80 backdrop-blur-sm flex items-center justify-center">
                      <div class="flex flex-col items-center gap-4">
                        <div class="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                        <p class="text-blue-400 font-mono text-xs tracking-widest animate-pulse">PROCESSING...</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Results Table -->
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
                    <th class="px-8 py-4 text-[10px] font-bold text-slate-400 uppercase tracking-widest">空间坐标</th>
                    <th class="px-8 py-4 text-[10px] font-bold text-slate-400 uppercase tracking-widest">状态</th>
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
                      <span class="text-[10px] font-mono text-slate-400 bg-slate-50 px-2 py-1 rounded-md">
                        {{ item.bbox.slice(0,2).join(', ') }} ...
                      </span>
                    </td>
                    <td class="px-8 py-5">
                      <span class="px-2 py-1 bg-blue-50 text-blue-600 text-[10px] font-black rounded uppercase tracking-tighter">Detected</span>
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
    </main>
  </div>
</template>

<style scoped>
/* 自定义滚动条 */
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

/* 移除 select 默认样式 */
select {
  appearance: none;
  background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2364748b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");
  background-repeat: no-repeat;
  background-position: right 0.75rem center;
  background-size: 1em;
  padding-right: 2.5rem;
}
</style>
