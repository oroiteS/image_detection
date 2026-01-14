<script setup lang="ts">
import { ref } from 'vue'

const show = ref(false)
const message = ref('')
const type = ref<'success' | 'error' | 'info'>('success')

const showToast = (msg: string, t: 'success' | 'error' | 'info' = 'success') => {
  message.value = msg
  type.value = t
  show.value = true
  setTimeout(() => {
    show.value = false
  }, 3000)
}

defineExpose({ showToast })
</script>

<template>
  <Transition name="toast">
    <div
      v-if="show"
      class="fixed top-6 left-1/2 -translate-x-1/2 z-[10000] px-6 py-3 rounded-full shadow-xl flex items-center gap-3 backdrop-blur-md border"
      :class="{
        'bg-green-500/90 border-green-400 text-white': type === 'success',
        'bg-red-500/90 border-red-400 text-white': type === 'error',
        'bg-blue-500/90 border-blue-400 text-white': type === 'info'
      }"
    >
      <span v-if="type === 'success'">✅</span>
      <span v-else-if="type === 'error'">❌</span>
      <span v-else>ℹ️</span>
      <span class="text-sm font-bold">{{ message }}</span>
    </div>
  </Transition>
</template>

<style scoped>
.toast-enter-active,
.toast-leave-active {
  transition: all 0.3s cubic-bezier(0.68, -0.55, 0.27, 1.55);
}

.toast-enter-from,
.toast-leave-to {
  opacity: 0;
  transform: translate(-50%, -20px);
}
</style>
