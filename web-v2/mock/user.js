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
    introduction: 'I am a super administrator',
    avatar: 'https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif',
    name: 'Super Admin'
  },
  'editor-token': {
    roles: ['editor'],
    introduction: 'I am an editor',
    avatar: 'https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif',
    name: 'Normal Editor'
  }
}

module.exports = [
  // user login
  {
    url: '/vue-element-admin/user/login',
    type: 'post',
    response: config => {
      const {username} = config.body
      const token = tokens[username]

      // mock error
      if (!token) {
        return {
          code: 60204,
          message: 'Account and password are incorrect.'
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
    url: '/vue-element-admin/user/info\.*',
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

  // user logout
  {
    url: '/vue-element-admin/user/logout',
    type: 'post',
    response: _ => {
      return {
        code: 20000,
        data: 'success'
      }
    }
  },

  // user register
  {
    url: '/vue-element-admin/user/register',
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
      const newToken = `${username}-token`
      const role = 'editor'
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


]
