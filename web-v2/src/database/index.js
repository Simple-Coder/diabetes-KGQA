import { initializeApp } from 'firebase/app'
import { getDatabase } from 'firebase/database'
import { getFirestore } from 'firebase/firestore'
import { getStorage } from 'firebase/storage'

// const config =
//   process.env.NODE_ENV === 'development'
//     ? process.env.VUE_APP_FIREBASE_CONFIG
//     // ? JSON.parse(process.env.VUE_APP_FIREBASE_CONFIG)
//     : JSON.parse(process.env.VUE_APP_FIREBASE_CONFIG_PUBLIC)
//
// console.log('Environment:', process.env.NODE_ENV)
// console.log('Firebase Config:', config)

const firebaseConfig = {
  'apiKey': 'AIzaSyDE90CPlCsl720t_bf-WE67gmkLu7C-Z6o',
  'authDomain': 'xd-chat-8f2fe.firebaseapp.com',
  'projectId': 'xd-chat-8f2fe',
  'storageBucket': 'xd-chat-8f2fe.appspot.com',
  'messagingSenderId': '564052539171',
  'appId': '1:564052539171:web:5c0c8e69072b64e2c47e96',
  'measurementId': 'G-FWMWL42CC0'
}

const app = initializeApp(firebaseConfig)

export const firestoreDb = getFirestore(app)
export const realtimeDb = getDatabase(app)
export const storage = getStorage(app)
