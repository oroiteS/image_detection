<script setup lang="ts">
import { ref } from 'vue'
import axios from 'axios'
import SHA256 from 'crypto-js/sha256'

const emit = defineEmits(['login-success', 'close'])

const isRegister = ref(false)
const username = ref('')
const password = ref('')
const confirmPassword = ref('')
const loading = ref(false)
const errorMsg = ref('')

const handleSubmit = async () => {
  if (!username.value || !password.value) {
    errorMsg.value = '请输入用户名和密码'
    return
  }

  if (isRegister.value && password.value !== confirmPassword.value) {
    errorMsg.value = '两次输入的密码不一致'
    return
  }

  loading.value = true
  errorMsg.value = ''

  try {
    // 前端哈希 (增加安全性，避免明文传输)
    const hashedPassword = SHA256(password.value).toString()

    const endpoint = isRegister.value ? 'http://127.0.0.1:8000/register' : 'http://127.0.0.1:8000/login'
    const res = await axios.post(endpoint, {
      username: username.value,
      password: hashedPassword
    })

    if (res.data.success) {
      if (isRegister.value) {
        // 注册成功后自动切换到登录
        isRegister.value = false
        errorMsg.value = ''
        alert('注册成功，请登录') // 这里可以用 Toast 替换，但为了简单先保留
      } else {
        localStorage.setItem('username', res.data.username)
        localStorage.setItem('token', res.data.token)
        emit('login-success', res.data.username)
      }
    }
  } catch (e: any) {
    errorMsg.value = e.response?.data?.detail || '请求失败，请检查网络'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="fixed inset-0 z-[100] flex items-center justify-center overflow-hidden">
    <!-- 背景 -->
    <div class="absolute inset-0 bg-slate-900">
      <div class="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80')] bg-cover bg-center opacity-20"></div>
      <div class="absolute inset-0 bg-gradient-to-b from-slate-900/50 to-slate-900"></div>
    </div>

    <!-- 登录卡片 -->
    <div class="relative w-full max-w-md p-8 mx-4 bg-white/10 backdrop-blur-xl border border-white/20 rounded-3xl shadow-2xl overflow-hidden animate-fade-in-up">
      <!-- 关闭按钮 -->
      <button
        @click="emit('close')"
        class="absolute top-4 right-4 text-white/50 hover:text-white transition-colors"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
      </button>

      <div class="text-center mb-8">
        <div class="w-16 h-16 bg-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg shadow-blue-500/30">
          <span class="text-3xl">⚡</span>
        </div>
        <h2 class="text-2xl font-bold text-white mb-1">
          {{ isRegister ? '创建新账号' : '欢迎回来' }}
        </h2>
        <p class="text-slate-400 text-sm">PowerAI 智能巡检系统</p>
      </div>

      <form @submit.prevent="handleSubmit" class="space-y-5">
        <div class="space-y-1">
          <label class="text-xs font-bold text-slate-300 uppercase ml-1">用户名</label>
          <div class="relative">
            <input
              v-model="username"
              type="text"
              class="w-full px-4 py-3 pl-10 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/30 focus:bg-white/10 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition-all"
              placeholder="请输入用户名"
            />
            <span class="absolute left-3 top-3.5 text-white/30">👤</span>
          </div>
        </div>

        <div class="space-y-1">
          <label class="text-xs font-bold text-slate-300 uppercase ml-1">密码</label>
          <div class="relative">
            <input
              v-model="password"
              type="password"
              class="w-full px-4 py-3 pl-10 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/30 focus:bg-white/10 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition-all"
              placeholder="请输入密码"
            />
            <span class="absolute left-3 top-3.5 text-white/30">🔒</span>
          </div>
        </div>

        <div v-if="isRegister" class="space-y-1 animate-slide-down">
          <label class="text-xs font-bold text-slate-300 uppercase ml-1">确认密码</label>
          <div class="relative">
            <input
              v-model="confirmPassword"
              type="password"
              class="w-full px-4 py-3 pl-10 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/30 focus:bg-white/10 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition-all"
              placeholder="请再次输入密码"
            />
            <span class="absolute left-3 top-3.5 text-white/30">🔐</span>
          </div>
        </div>

        <div v-if="errorMsg" class="text-red-400 text-xs font-bold text-center bg-red-500/10 border border-red-500/20 py-2 rounded-lg animate-shake">
          {{ errorMsg }}
        </div>

        <button
          type="submit"
          :disabled="loading"
          class="w-full py-3.5 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl transition-all shadow-lg shadow-blue-600/30 flex justify-center items-center group"
        >
          <span v-if="loading" class="animate-spin mr-2">⏳</span>
          <span v-else class="group-hover:scale-105 transition-transform">
            {{ isRegister ? '立即注册' : '登录系统' }}
          </span>
        </button>
      </form>

      <div class="mt-8 text-center text-sm text-slate-400">
        {{ isRegister ? '已有账号？' : '还没有账号？' }}
        <button @click="isRegister = !isRegister" class="text-white font-bold hover:text-blue-400 transition-colors ml-1">
          {{ isRegister ? '去登录' : '去注册' }}
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.animate-fade-in-up {
  animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}

.animate-slide-down {
  animation: slideDown 0.3s ease-out;
}

.animate-shake {
  animation: shake 0.5s cubic-bezier(.36,.07,.19,.97) both;
}

@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes slideDown {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes shake {
  10%, 90% { transform: translate3d(-1px, 0, 0); }
  20%, 80% { transform: translate3d(2px, 0, 0); }
  30%, 50%, 70% { transform: translate3d(-4px, 0, 0); }
  40%, 60% { transform: translate3d(4px, 0, 0); }
}
</style>
