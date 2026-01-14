<script setup lang="ts">
import { ref, watch } from 'vue'

const props = defineProps<{
  src: string | null
}>()

const emit = defineEmits(['close'])

const scale = ref(1)
const translateX = ref(0)
const translateY = ref(0)
const isDragging = ref(false)
const startX = ref(0)
const startY = ref(0)

// 当图片变化时重置状态
watch(() => props.src, () => {
  scale.value = 1
  translateX.value = 0
  translateY.value = 0
})

const zoomIn = () => {
  scale.value = Math.min(scale.value + 0.25, 5)
}

const zoomOut = () => {
  scale.value = Math.max(scale.value - 0.25, 0.5)
}

const resetZoom = () => {
  scale.value = 1
  translateX.value = 0
  translateY.value = 0
}

const onWheel = (e: WheelEvent) => {
  e.preventDefault()
  if (e.deltaY < 0) {
    zoomIn()
  } else {
    zoomOut()
  }
}

const startDrag = (e: MouseEvent) => {
  if (scale.value <= 1) return // 只有放大后才允许拖拽
  isDragging.value = true
  startX.value = e.clientX - translateX.value
  startY.value = e.clientY - translateY.value
}

const onDrag = (e: MouseEvent) => {
  if (!isDragging.value) return
  e.preventDefault()
  translateX.value = e.clientX - startX.value
  translateY.value = e.clientY - startY.value
}

const stopDrag = () => {
  isDragging.value = false
}
</script>

<template>
  <Transition name="fade">
    <div
      v-if="src"
      class="fixed inset-0 z-[9999] bg-black/95 backdrop-blur-md flex flex-col items-center justify-center overflow-hidden select-none"
      @click="emit('close')"
      @wheel="onWheel"
    >
      <!-- 图片容器 -->
      <div
        class="relative w-full h-full flex items-center justify-center"
        @mousedown.stop="startDrag"
        @mousemove="onDrag"
        @mouseup="stopDrag"
        @mouseleave="stopDrag"
      >
        <img
          :src="src"
          class="max-w-full max-h-full object-contain transition-transform duration-100 ease-out"
          :style="{
            transform: `translate(${translateX}px, ${translateY}px) scale(${scale})`,
            cursor: scale > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default'
          }"
          @click.stop
          draggable="false"
        />
      </div>

      <!-- 关闭按钮 -->
      <button
        class="absolute top-6 right-6 w-10 h-10 flex items-center justify-center rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors z-50"
        @click="emit('close')"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
      </button>

      <!-- 底部控制栏 -->
      <div
        class="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-4 px-6 py-3 bg-white/10 backdrop-blur-md rounded-full border border-white/10 z-50"
        @click.stop
      >
        <button @click="zoomOut" class="p-2 text-white/80 hover:text-white hover:bg-white/10 rounded-full transition-colors" title="缩小">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line><line x1="8" y1="11" x2="14" y2="11"></line></svg>
        </button>

        <span class="text-sm font-mono font-bold text-white w-12 text-center">{{ (scale * 100).toFixed(0) }}%</span>

        <button @click="zoomIn" class="p-2 text-white/80 hover:text-white hover:bg-white/10 rounded-full transition-colors" title="放大">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line><line x1="11" y1="8" x2="11" y2="14"></line><line x1="8" y1="11" x2="14" y2="11"></line></svg>
        </button>

        <div class="w-px h-4 bg-white/20 mx-2"></div>

        <button @click="resetZoom" class="p-2 text-white/80 hover:text-white hover:bg-white/10 rounded-full transition-colors" title="重置">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path><path d="M3 3v5h5"></path></svg>
        </button>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
