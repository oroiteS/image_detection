import './style.css'
import 'viewerjs/dist/viewer.css'
import { createApp } from 'vue'
import VueViewer from 'v-viewer'
import App from './App.vue'

const app = createApp(App)
app.use(VueViewer)
app.mount('#app')
