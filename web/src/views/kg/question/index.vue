<template>
  <el-card class="box-card">
    <template #header>
      <span style="font-weight: bold;font-size: 15px">基于知识图谱的糖尿病问答系统</span>
    </template>
    <div id="dialog_container">
      <div v-for="oneDialog in text_dialog" :key="oneDialog">
        <el-divider content-position="left">{{ user_name }} --{{ oneDialog.time }}</el-divider>
        <span id="question_card" style="font-size: 15px">{{ oneDialog.question }}</span>
        <el-divider content-position="right">回答</el-divider>
        <span id="answer_card">
              <div style="font-size: 15px" v-html="oneDialog.answer"></div>
            </span>
      </div>
    </div>
    <el-divider content-position="right"></el-divider>
    <el-input
      type="textarea"
      :autosize="{ minRows: 2, maxRows: 4}"
      placeholder="尝试输入，糖尿病相关，如：糖尿病的临床表现？糖尿病如何治疗？"
      v-model="txt_question"
    >
    </el-input>
    <el-divider content-position="right">
      <el-button @click="ask_question()">提问</el-button>
    </el-divider>
  </el-card>
</template>

<script>
import {doAnswer} from '@/api/answer'

export default {
  name: 'AnswerCard',
  methods: {
    scrollToBottom: function () {
      // 问答的框，每次提问完滚动条滚动到最下方新的消息
      this.$nextTick(() => {
        const div = document.getElementById('dialog_container')
        div.scrollTop = div.scrollHeight
      })

    },
    ask_question() {
      // 提问
      if (this.txt_question === '') {
        alert("输入不能为空")
        return
      }
      const params = {
        'text': this.txt_question
      }
      console.log(params)

      const question = this.txt_question
      // 添加一条 问答对话
      const myDate = new Date();
      let answerText = '我是一条答案'
      doAnswer(params).then((res) => {
        console.log('响应成功')
        console.log(res)
        answerText = res.data.reply
        this.text_dialog.push({time: myDate.toLocaleString(), question: question, answer: answerText})
      }).catch(error => {
        console.log('响应失败')
        console.log(error)
        this.text_dialog.push({time: myDate.toLocaleString(), question: question, answer: '系统异常'})
      });

      this.scrollToBottom();
      this.txt_question = ''
    }
  },
  data() {
    return {
      user_name: '默认用户',
      txt_question: '',
      text_dialog: [],
    }
  }
}
</script>

<style scoped>


.box-card {
  margin: 2% auto;
  width: 50%;
  min-width: 900px;
  text-align: left;
}

#dialog_container {
  overflow: auto;
  scroll-margin-right: 1px;
  /*根据屏幕占比设置高度*/
  min-height: calc(100vh - 360px);
  max-height: calc(100vh - 360px);
}
</style>
