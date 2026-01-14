<script setup lang="ts">
import { ref, onMounted } from 'vue'
import axios from 'axios'
import Lightbox from './Lightbox.vue'

const props = defineProps<{
  username: string
}>()

interface HistoryItem {
  id: number
  filename: string
  model_name: string
  inference_time: number
  detection_count: number
  result_image_base64: string
  timestamp: string
}

const history = ref<HistoryItem[]>([])
const loading = ref(false)
const lightboxImage = ref<string | null>(null)

const fetchHistory = async () => {
  loading.value = true
  try {
    const res = await axios.get(`http://127.0.0.1:8000/history?username=${props.username}`)
    history.value = res.data
  } catch (e) {
    console.error('获取历史记录失败', e)
  } finally {
    loading.value = false
  }
}

onMounted(fetchHistory)
</script>

<template>
  <div class="space-y-6">
    <Lightbox :src="lightboxImage" @close="lightboxImage = null" />

    <div class="flex justify-between items-center">
      <h2 class="text-xl font-bold text-slate-800">检测历史记录</h2>
      <button @click="fetchHistory" class="text-sm text-blue-600 hover:underline">刷新列表</button>
    </div>

    <div v-if="loading" class="text-center py-12 text-slate-400">加载中...</div>

    <div v-else-if="history.length === 0" class="text-center py-12 text-slate-400 bg-white rounded-2xl border border-slate-200">
      暂无历史记录
    </div>

    <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      <div v-for="item in history" :key="item.id" class="bg-white rounded-2xl border border-slate-200 overflow-hidden hover:shadow-lg transition-shadow">
        <div
          class="aspect-video bg-slate-100 relative group cursor-zoom-in"
          @click="lightboxImage = item.result_image_base64"
        >
          <img :src="item.result_image_base64" class="w-full h-full object-cover" />
          <div class="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors"></div>
        </div>
        <div class="p-4">
          <div class="flex justify-between items-start mb-2">
            <h3 class="font-bold text-slate-700 truncate pr-2" :title="item.filename">{{ item.filename }}</h3>
            <span class="text-xs bg-slate-100 text-slate-500 px-2 py-1 rounded">{{ item.timestamp.split(' ')[0] }}</span>
          </div>
          <div class="space-y-1 text-xs text-slate-500">
            <div class="flex justify-between">
              <span>模型:</span>
              <span class="font-mono text-slate-700">{{ item.model_name }}</span>
            </div>
            <div class="flex justify-between">
              <span>耗时:</span>
              <span class="font-mono text-blue-600">{{ item.inference_time.toFixed(1) }}ms</span>
            </div>
            <div class="flex justify-between">
              <span>缺陷数:</span>
              <span class="font-mono text-red-500 font-bold">{{ item.detection_count }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
