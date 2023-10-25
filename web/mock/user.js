const tokens = {
  admin: {
    token: 'admin-token'
  },
  editor: {
    token: 'editor-token'
  }
}

const users = {
  'admin-token': {
    roles: ['admin'],
    roleNames: ['超级管理员'],
    introduction: 'I am a super administrator',
    avatar: 'https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif',
    name: 'admin',
    pwd: 123456
  },
  'editor-token': {
    roles: ['common'],
    roleNames: ['普通用户'],
    introduction: 'I am an editor',
    avatar: 'https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif',
    name: 'common',
    pwd: 123456
  }
}

module.exports = [
  // user login
  {
    url: '/vue-admin-template/user/login',
    type: 'post',
    response: config => {
      const {username} = config.body
      const token = tokens[username]

      // mock error
      if (!token) {
        return {
          code: 60204,
          message: '用户名或密码不正确'
        }
      }


      return {
        code: 20000,
        data: token
      }
    }
  },

  // get user info
  {
    url: '/vue-admin-template/user/info\.*',
    type: 'get',
    response: config => {
      const {token} = config.query
      const info = users[token]

      // mock error
      if (!info) {
        return {
          code: 50008,
          message: 'Login failed, unable to get user details.'
        }
      }

      return {
        code: 20000,
        data: info
      }
    }
  },

  // get all users
  {
    url: '/vue-admin-template/user/all',
    type: 'get',
    response: () => {
      // Convert the 'users' object into an array of user objects
      const allUsers = Object.keys(users).map(token => users[token]);

      return {
        code: 20000,
        data: allUsers
      };
    }
  },


  // user register
  {
    url: '/vue-admin-template/user/register',
    type: 'post',
    response: config => {
      const {username, password} = config.body;
      // const {username, password, role} = config.body;

      // Check if the username is already taken
      if (tokens[username]) {
        return {
          code: 60205,
          message: '用户名已存在！请重新输入用户名!'
        };
      }

      // Generate a new token for the registered user
      const newToken = `${username}-token`;
      const role = 'common'
      const roleName = '普通用户'
      // Create a new user entry
      users[newToken] = {
        roles: [role],
        roleNames: [roleName],
        introduction: `I am a ${role}`,
        avatar: 'https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif',
        name: username,
        pwd: password
      };

      // Add the new user's token to the tokens object
      tokens[username] = {
        token: newToken
      };

      return {
        code: 20000,
        data: newToken
      };
    }
  },

  // user logout
  {
    url: '/vue-admin-template/user/logout',
    type: 'post',
    response: _ => {
      return {
        code: 20000,
        data: 'success'
      }
    }
  }
]
