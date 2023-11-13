<template>
  <div class="app-container">
    <el-form ref="queryForm" :model="queryParams" size="small" :inline="true" label-width="68px">
      <el-form-item label="登录地址" prop="ipaddr">
        <el-input
          v-model="queryParams.ipaddr"
          placeholder="请输入登录地址"
          clearable
          @keyup.enter.native="handleQuery"
        />
      </el-form-item>
      <el-form-item label="用户名称" prop="userName">
        <el-input
          v-model="queryParams.userName"
          placeholder="请输入用户名称"
          clearable
          @keyup.enter.native="handleQuery"
        />
      </el-form-item>
      <el-form-item>
        <el-button type="primary" icon="el-icon-search" size="mini" @click="handleQuery">搜索</el-button>
        <el-button icon="el-icon-refresh" size="mini" @click="resetQuery">重置</el-button>
      </el-form-item>

    </el-form>
    <el-table
      v-loading="loading"
      :data="list.slice((pageNum-1)*pageSize,pageNum*pageSize)"
      style="width: 100%;"
    >
      <el-table-column label="序号" type="index" align="center">
        <template slot-scope="scope">
          <span>{{ (pageNum - 1) * pageSize + scope.$index + 1 }}</span>
        </template>
      </el-table-column>
      <el-table-column label="会话编号" align="center" prop="tokenId" :show-overflow-tooltip="true" />
      <el-table-column label="登录名称" align="center" prop="userName" :show-overflow-tooltip="true" />
      <el-table-column label="部门名称" align="center" prop="deptName" />
      <el-table-column label="主机" align="center" prop="ipaddr" :show-overflow-tooltip="true" />
      <el-table-column label="登录地点" align="center" prop="loginLocation" :show-overflow-tooltip="true" />
      <el-table-column label="浏览器" align="center" prop="browser" />
      <el-table-column label="操作系统" align="center" prop="os" />
      <el-table-column label="登录时间" align="center" prop="loginTime" width="180">
        <template slot-scope="scope">
          <span>{{ parseTime(scope.row.loginTime) }}</span>
        </template>
      </el-table-column>
      <el-table-column label="操作" align="center" class-name="small-padding fixed-width">
        <template slot-scope="scope">
          <el-button
            size="mini"
            type="text"
            icon="el-icon-delete"
            @click="handleForceLogout(scope.row)"
          >强退
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <pagination v-show="total>0" :total="total" :page.sync="pageNum" :limit.sync="pageSize" />
  </div>
</template>

<script>
// import { list, forceLogout } from "@/api/monitor/online";
import { WebSocketModule } from '@/assets/js/mywebsoket-v2'
import { parseTimestamp, formatTimestamp } from '@/utils/dates'
import * as firestoreService from '@/database/firestore'

export default {
  name: 'Online',
  data() {
    return {
      currentUserId: 'admin',
      // 遮罩层
      loading: true,
      // 总条数
      total: 0,
      // 表格数据
      list: [],
      pageNum: 1,
      pageSize: 10,
      // 查询参数
      queryParams: {
        ipaddr: undefined,
        userName: undefined
      }
    }
  },
  created() {
    // this.getList();
    this.getListTest()
    // this.initWebSocket()
  },
  methods: {
    getListTest() {
      this.loading = true
      this.list = [
        {
          'tokenId': '8855e2e0-2bff-4250-9d9e-7dfd7ee5a1e9',
          'deptName': '研发部门',
          'userName': 'admin',
          'ipaddr': '182.137.106.29',
          'loginLocation': '四川省 绵阳市',
          'browser': 'Chrome 10',
          'os': 'Windows 10',
          'loginTime': 1699855873385
        }
      ]
      this.total = 1
      this.loading = false
      /* list(this.queryParams).then(response => {
         this.list = response.rows;
         this.total = response.total;
         this.loading = false;
       });*/
    },
    /** 查询登录日志列表 */
    /*    getList() {
      this.loading = true;
      list(this.queryParams).then(response => {
        this.list = response.rows;
        this.total = response.total;
        this.loading = false;
      });
    },
    /!** 搜索按钮操作 *!/
    handleQuery() {
      this.pageNum = 1;
      this.getList();
    },
    /!** 强退按钮操作 *!/
    handleForceLogout(row) {
      this.$modal.confirm('是否确认强退名称为"' + row.userName + '"的用户？').then(function() {
        return forceLogout(row.tokenId);
      }).then(() => {
        this.getList();
        this.$modal.msgSuccess("强退成功");
      }).catch(() => {});
    }*/
    handleQuery() {
      this.pageNum = 1
      this.getListTest()
    },
    resetQuery() {
      this.resetForm('queryForm')
      this.handleQuery()
    },

    // 以下是新增的websocket
    isValidJSON(str) {
      try {
        JSON.parse(str)
        return true
      } catch (error) {
        return false
      }
    },
    initWebSocket() {
      this.webSocketModule = new WebSocketModule(this.currentUserId, this.wsuri, this.websockonmessage)
      this.webSocketModule.connect()
    },
    websockonmessage(e) {
      console.log(e)

      /*
      *
      * class MsgType:
    Login_Up = 1
    Login_Down = 2
    GetAllUserIds_Up = 3
    GetAllUserIds_Down = 4
    ASK_Up = 99
    ASK_Down = 98

      *
      *
      * */
      const message = JSON.parse(e.data)
      const msgType = message.type
      if (msgType === undefined ||
        msgType !== 4) {
        console.log('未知消息', message)
        return
      }

      this.list = message.data
      this.total = message.length
      this.loading = false
    },
    sendWsMessage() {
      try {
        if (!this.webSocketModule.isConnected) {
          console.log('连接异常')
          // this.dialogData.push({person: this.input, rot: '服务异常，请稍后再试！'})
          // this.input = ''
          return
        }
        const params = this.queryParams
        console.log('准备发送ws online userId')
        console.log(params)
        const sendData = JSON.stringify(params)
        this.webSocketModule.send(sendData)
      } catch (e) {
        console.log(e)
        console.log(' online ws发送异常！')
      }
    }
  }
}
</script>
