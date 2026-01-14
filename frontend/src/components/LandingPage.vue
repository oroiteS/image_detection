<script setup lang="ts">
const emit = defineEmits(['start'])
</script>

<template>
  <div class="fixed inset-0 z-50 bg-slate-900 text-white flex flex-col items-center justify-center overflow-hidden">
    <!-- 背景动画 (简单的光晕效果) -->
    <div class="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
      <div class="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/20 rounded-full blur-[120px] animate-pulse"></div>
      <div class="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/20 rounded-full blur-[120px] animate-pulse" style="animation-delay: 2s"></div>
    </div>

    <!-- 内容区域 -->
    <div class="relative z-10 text-center max-w-4xl px-6">
      <div class="mb-8 inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 backdrop-blur-sm">
        <span class="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
        <span class="text-xs font-bold tracking-widest text-slate-300 uppercase">System Online</span>
      </div>

      <h1 class="text-5xl md:text-7xl font-black tracking-tight mb-6 bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
        电力巡检<br/>
        <span class="text-blue-500">智能检测系统</span>
      </h1>

      <p class="text-lg md:text-xl text-slate-400 mb-12 max-w-2xl mx-auto leading-relaxed">
        基于 <span class="text-white font-bold">YOLOv11</span> 与 <span class="text-white font-bold">TensorRT</span> 的高性能 AI 解决方案。
        支持毫秒级实时推理、多模型对比分析及误检反馈闭环。
      </p>

      <div class="flex flex-col md:flex-row items-center justify-center gap-6">
        <button
          @click="emit('start')"
          class="group relative px-8 py-4 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl transition-all shadow-[0_0_20px_rgba(37,99,235,0.3)] hover:shadow-[0_0_30px_rgba(37,99,235,0.5)] hover:-translate-y-1"
        >
          <span class="relative z-10 flex items-center gap-2">
            立即开始
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="group-hover:translate-x-1 transition-transform"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
          </span>
        </button>

        <a href="https://github.com/oroiteS/image_detection" target="_blank" class="px-8 py-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl font-bold transition-all flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
          查看源码
        </a>
      </div>

      <!-- 特性网格 -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mt-20 text-left">
        <div class="p-6 rounded-2xl bg-white/5 border border-white/5 hover:bg-white/10 transition-colors">
          <div class="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4 text-blue-400 text-xl">🚀</div>
          <h3 class="font-bold text-white mb-2">极致性能</h3>
          <p class="text-sm text-slate-400">集成 TensorRT C++ 推理引擎，支持 INT8 量化加速，推理速度提升 5-10 倍。</p>
        </div>
        <div class="p-6 rounded-2xl bg-white/5 border border-white/5 hover:bg-white/10 transition-colors">
          <div class="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center mb-4 text-purple-400 text-xl">📊</div>
          <h3 class="font-bold text-white mb-2">深度分析</h3>
          <p class="text-sm text-slate-400">支持多模型同屏对比，实时查看置信度与耗时差异，快速定位模型短板。</p>
        </div>
        <div class="p-6 rounded-2xl bg-white/5 border border-white/5 hover:bg-white/10 transition-colors">
          <div class="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center mb-4 text-green-400 text-xl">🔄</div>
          <h3 class="font-bold text-white mb-2">闭环优化</h3>
          <p class="text-sm text-slate-400">内置误检/漏检反馈机制，自动收集难例数据，助力模型持续迭代优化。</p>
        </div>
      </div>
    </div>

    <!-- 底部版权 -->
    <div class="absolute bottom-6 text-xs text-slate-600">
      &copy; 2024 PowerAI Inspection System. All rights reserved.
    </div>
  </div>
</template>
