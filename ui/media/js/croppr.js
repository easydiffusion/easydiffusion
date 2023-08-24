/**
 * Croppr.js
 * https://github.com/jamesssooi/Croppr.js
 * 
 * A JavaScript image cropper that's lightweight, awesome, and has
 * zero dependencies.
 * 
 * (C) 2017 James Ooi. Released under the MIT License.
 */

(function (global, factory) {
	typeof exports === 'object' && typeof module !== 'undefined' ? module.exports = factory() :
	typeof define === 'function' && define.amd ? define(factory) :
	(global.Croppr = factory());
}(this, (function () { 'use strict';

(function () {
  var lastTime = 0;
  var vendors = ['ms', 'moz', 'webkit', 'o'];
  for (var x = 0; x < vendors.length && !window.requestAnimationFrame; ++x) {
    window.requestAnimationFrame = window[vendors[x] + 'RequestAnimationFrame'];
    window.cancelAnimationFrame = window[vendors[x] + 'CancelAnimationFrame'] || window[vendors[x] + 'CancelRequestAnimationFrame'];
  }
  if (!window.requestAnimationFrame) window.requestAnimationFrame = function (callback, element) {
    var currTime = new Date().getTime();
    var timeToCall = Math.max(0, 16 - (currTime - lastTime));
    var id = window.setTimeout(function () {
      callback(currTime + timeToCall);
    }, timeToCall);
    lastTime = currTime + timeToCall;
    return id;
  };
  if (!window.cancelAnimationFrame) window.cancelAnimationFrame = function (id) {
    clearTimeout(id);
  };
})();
(function () {
  if (typeof window.CustomEvent === "function") return false;
  function CustomEvent(event, params) {
    params = params || { bubbles: false, cancelable: false, detail: undefined };
    var evt = document.createEvent('CustomEvent');
    evt.initCustomEvent(event, params.bubbles, params.cancelable, params.detail);
    return evt;
  }
  CustomEvent.prototype = window.Event.prototype;
  window.CustomEvent = CustomEvent;
})();
(function (window) {
  try {
    new CustomEvent('test');
    return false;
  } catch (e) {}
  function MouseEvent(eventType, params) {
    params = params || { bubbles: false, cancelable: false };
    var mouseEvent = document.createEvent('MouseEvent');
    mouseEvent.initMouseEvent(eventType, params.bubbles, params.cancelable, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
    return mouseEvent;
  }
  MouseEvent.prototype = Event.prototype;
  window.MouseEvent = MouseEvent;
})(window);

var classCallCheck = function (instance, Constructor) {
  if (!(instance instanceof Constructor)) {
    throw new TypeError("Cannot call a class as a function");
  }
};

var createClass = function () {
  function defineProperties(target, props) {
    for (var i = 0; i < props.length; i++) {
      var descriptor = props[i];
      descriptor.enumerable = descriptor.enumerable || false;
      descriptor.configurable = true;
      if ("value" in descriptor) descriptor.writable = true;
      Object.defineProperty(target, descriptor.key, descriptor);
    }
  }

  return function (Constructor, protoProps, staticProps) {
    if (protoProps) defineProperties(Constructor.prototype, protoProps);
    if (staticProps) defineProperties(Constructor, staticProps);
    return Constructor;
  };
}();







var get = function get(object, property, receiver) {
  if (object === null) object = Function.prototype;
  var desc = Object.getOwnPropertyDescriptor(object, property);

  if (desc === undefined) {
    var parent = Object.getPrototypeOf(object);

    if (parent === null) {
      return undefined;
    } else {
      return get(parent, property, receiver);
    }
  } else if ("value" in desc) {
    return desc.value;
  } else {
    var getter = desc.get;

    if (getter === undefined) {
      return undefined;
    }

    return getter.call(receiver);
  }
};

var inherits = function (subClass, superClass) {
  if (typeof superClass !== "function" && superClass !== null) {
    throw new TypeError("Super expression must either be null or a function, not " + typeof superClass);
  }

  subClass.prototype = Object.create(superClass && superClass.prototype, {
    constructor: {
      value: subClass,
      enumerable: false,
      writable: true,
      configurable: true
    }
  });
  if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass;
};











var possibleConstructorReturn = function (self, call) {
  if (!self) {
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  }

  return call && (typeof call === "object" || typeof call === "function") ? call : self;
};





var slicedToArray = function () {
  function sliceIterator(arr, i) {
    var _arr = [];
    var _n = true;
    var _d = false;
    var _e = undefined;

    try {
      for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) {
        _arr.push(_s.value);

        if (i && _arr.length === i) break;
      }
    } catch (err) {
      _d = true;
      _e = err;
    } finally {
      try {
        if (!_n && _i["return"]) _i["return"]();
      } finally {
        if (_d) throw _e;
      }
    }

    return _arr;
  }

  return function (arr, i) {
    if (Array.isArray(arr)) {
      return arr;
    } else if (Symbol.iterator in Object(arr)) {
      return sliceIterator(arr, i);
    } else {
      throw new TypeError("Invalid attempt to destructure non-iterable instance");
    }
  };
}();

var Handle =
/**
 * Creates a new Handle instance.
 * @constructor
 * @param {Array} position The x and y ratio position of the handle
 *      within the crop region. Accepts a value between 0 to 1 in the order
 *      of [X, Y].
 * @param {Array} constraints Define the side of the crop region that
 *      is to be affected by this handle. Accepts a value of 0 or 1 in the
 *      order of [TOP, RIGHT, BOTTOM, LEFT].
 * @param {String} cursor The CSS cursor of this handle.
 * @param {Element} eventBus The element to dispatch events to.
 */
function Handle(position, constraints, cursor, eventBus) {
  classCallCheck(this, Handle);
  var self = this;
  this.position = position;
  this.constraints = constraints;
  this.cursor = cursor;
  this.eventBus = eventBus;
  this.el = document.createElement('div');
  this.el.className = 'croppr-handle';
  this.el.style.cursor = cursor;
  this.el.addEventListener('mousedown', onMouseDown);
  function onMouseDown(e) {
    e.stopPropagation();
    document.addEventListener('mouseup', onMouseUp);
    document.addEventListener('mousemove', onMouseMove);
    self.eventBus.dispatchEvent(new CustomEvent('handlestart', {
      detail: { handle: self }
    }));
  }
  function onMouseUp(e) {
    e.stopPropagation();
    document.removeEventListener('mouseup', onMouseUp);
    document.removeEventListener('mousemove', onMouseMove);
    self.eventBus.dispatchEvent(new CustomEvent('handleend', {
      detail: { handle: self }
    }));
  }
  function onMouseMove(e) {
    e.stopPropagation();
    self.eventBus.dispatchEvent(new CustomEvent('handlemove', {
      detail: { mouseX: e.clientX, mouseY: e.clientY }
    }));
  }
};

var Box = function () {
  /**
   * Creates a new Box instance.
   * @constructor
   * @param {Number} x1
   * @param {Number} y1
   * @param {Number} x2
   * @param {Number} y2
   */
  function Box(x1, y1, x2, y2) {
    classCallCheck(this, Box);
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
  }
  /**
   * Sets the new dimensions of the box.
   * @param {Number} x1
   * @param {Number} y1
   * @param {Number} x2
   * @param {Number} y2
   */
  createClass(Box, [{
    key: 'set',
    value: function set$$1() {
      var x1 = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : null;
      var y1 = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : null;
      var x2 = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : null;
      var y2 = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : null;
      this.x1 = x1 == null ? this.x1 : x1;
      this.y1 = y1 == null ? this.y1 : y1;
      this.x2 = x2 == null ? this.x2 : x2;
      this.y2 = y2 == null ? this.y2 : y2;
      return this;
    }
    /**
     * Calculates the width of the box.
     * @returns {Number}
     */
  }, {
    key: 'width',
    value: function width() {
      return Math.abs(this.x2 - this.x1);
    }
    /**
     * Calculates the height of the box.
     * @returns {Number}
     */
  }, {
    key: 'height',
    value: function height() {
      return Math.abs(this.y2 - this.y1);
    }
    /**
     * Resizes the box to a new size.
     * @param {Number} newWidth
     * @param {Number} newHeight
     * @param {Array} [origin] The origin point to resize from.
     *      Defaults to [0, 0] (top left).
     */
  }, {
    key: 'resize',
    value: function resize(newWidth, newHeight) {
      var origin = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : [0, 0];
      var fromX = this.x1 + this.width() * origin[0];
      var fromY = this.y1 + this.height() * origin[1];
      this.x1 = fromX - newWidth * origin[0];
      this.y1 = fromY - newHeight * origin[1];
      this.x2 = this.x1 + newWidth;
      this.y2 = this.y1 + newHeight;
      return this;
    }
    /**
     * Scale the box by a factor.
     * @param {Number} factor
     * @param {Array} [origin] The origin point to resize from.
     *      Defaults to [0, 0] (top left).
     */
  }, {
    key: 'scale',
    value: function scale(factor) {
      var origin = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : [0, 0];
      var newWidth = this.width() * factor;
      var newHeight = this.height() * factor;
      this.resize(newWidth, newHeight, origin);
      return this;
    }
  }, {
    key: 'move',
    value: function move() {
      var x = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : null;
      var y = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : null;
      var width = this.width();
      var height = this.height();
      x = x === null ? this.x1 : x;
      y = y === null ? this.y1 : y;
      this.x1 = x;
      this.y1 = y;
      this.x2 = x + width;
      this.y2 = y + height;
      return this;
    }
    /**
     * Get relative x and y coordinates of a given point within the box.
     * @param {Array} point The x and y ratio position within the box.
     * @returns {Array} The x and y coordinates [x, y].
     */
  }, {
    key: 'getRelativePoint',
    value: function getRelativePoint() {
      var point = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [0, 0];
      var x = this.width() * point[0];
      var y = this.height() * point[1];
      return [x, y];
    }
    /**
     * Get absolute x and y coordinates of a given point within the box.
     * @param {Array} point The x and y ratio position within the box.
     * @returns {Array} The x and y coordinates [x, y].
     */
  }, {
    key: 'getAbsolutePoint',
    value: function getAbsolutePoint() {
      var point = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [0, 0];
      var x = this.x1 + this.width() * point[0];
      var y = this.y1 + this.height() * point[1];
      return [x, y];
    }
    /**
     * Constrain the box to a fixed ratio.
     * @param {Number} ratio
     * @param {Array} [origin] The origin point to resize from.
     *     Defaults to [0, 0] (top left).
     * @param {String} [grow] The axis to grow to maintain the ratio.
     *     Defaults to 'height'.
     */
  }, {
    key: 'constrainToRatio',
    value: function constrainToRatio(ratio) {
      var origin = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : [0, 0];
      var grow = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : 'height';
      if (ratio === null) {
        return;
      }
      var width = this.width();
      var height = this.height();
      switch (grow) {
        case 'height':
          this.resize(this.width(), this.width() * ratio, origin);
          break;
        case 'width':
          this.resize(this.height() * 1 / ratio, this.height(), origin);
          break;
        default:
          this.resize(this.width(), this.width() * ratio, origin);
      }
      return this;
    }
    /**
     * Constrain the box within a boundary.
     * @param {Number} boundaryWidth
     * @param {Number} boundaryHeight
     * @param {Array} [origin] The origin point to resize from.
     *     Defaults to [0, 0] (top left).
     */
  }, {
    key: 'constrainToBoundary',
    value: function constrainToBoundary(boundaryWidth, boundaryHeight) {
      var origin = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : [0, 0];
      var _getAbsolutePoint = this.getAbsolutePoint(origin),
          _getAbsolutePoint2 = slicedToArray(_getAbsolutePoint, 2),
          originX = _getAbsolutePoint2[0],
          originY = _getAbsolutePoint2[1];
      var maxIfLeft = originX;
      var maxIfTop = originY;
      var maxIfRight = boundaryWidth - originX;
      var maxIfBottom = boundaryHeight - originY;
      var directionX = -2 * origin[0] + 1;
      var directionY = -2 * origin[1] + 1;
      var maxWidth = null,
          maxHeight = null;
      switch (directionX) {
        case -1:
          maxWidth = maxIfLeft;break;
        case 0:
          maxWidth = Math.min(maxIfLeft, maxIfRight) * 2;break;
        case +1:
          maxWidth = maxIfRight;break;
      }
      switch (directionY) {
        case -1:
          maxHeight = maxIfTop;break;
        case 0:
          maxHeight = Math.min(maxIfTop, maxIfBottom) * 2;break;
        case +1:
          maxHeight = maxIfBottom;break;
      }
      if (this.width() > maxWidth) {
        var factor = maxWidth / this.width();
        this.scale(factor, origin);
      }
      if (this.height() > maxHeight) {
        var _factor = maxHeight / this.height();
        this.scale(_factor, origin);
      }
      return this;
    }
    /**
     * Constrain the box to a maximum/minimum size.
     * @param {Number} [maxWidth]
     * @param {Number} [maxHeight]
     * @param {Number} [minWidth]
     * @param {Number} [minHeight]
     * @param {Array} [origin] The origin point to resize from.
     *     Defaults to [0, 0] (top left).
     * @param {Number} [ratio] Ratio to maintain.
     */
  }, {
    key: 'constrainToSize',
    value: function constrainToSize() {
      var maxWidth = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : null;
      var maxHeight = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : null;
      var minWidth = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : null;
      var minHeight = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : null;
      var origin = arguments.length > 4 && arguments[4] !== undefined ? arguments[4] : [0, 0];
      var ratio = arguments.length > 5 && arguments[5] !== undefined ? arguments[5] : null;
      if (ratio) {
        if (ratio > 1) {
          maxWidth = maxHeight * 1 / ratio;
          minHeight = minHeight * ratio;
        } else if (ratio < 1) {
          maxHeight = maxWidth * ratio;
          minWidth = minHeight * 1 / ratio;
        }
      }
      if (maxWidth && this.width() > maxWidth) {
        var newWidth = maxWidth,
            newHeight = ratio === null ? this.height() : maxHeight;
        this.resize(newWidth, newHeight, origin);
      }
      if (maxHeight && this.height() > maxHeight) {
        var _newWidth = ratio === null ? this.width() : maxWidth,
            _newHeight = maxHeight;
        this.resize(_newWidth, _newHeight, origin);
      }
      if (minWidth && this.width() < minWidth) {
        var _newWidth2 = minWidth,
            _newHeight2 = ratio === null ? this.height() : minHeight;
        this.resize(_newWidth2, _newHeight2, origin);
      }
      if (minHeight && this.height() < minHeight) {
        var _newWidth3 = ratio === null ? this.width() : minWidth,
            _newHeight3 = minHeight;
        this.resize(_newWidth3, _newHeight3, origin);
      }
      return this;
    }
  }]);
  return Box;
}();

/**
 * Binds an element's touch events to be simulated as mouse events.
 * @param {Element} element
 */
function enableTouch(element) {
  element.addEventListener('touchstart', simulateMouseEvent);
  element.addEventListener('touchend', simulateMouseEvent);
  element.addEventListener('touchmove', simulateMouseEvent);
}
/**
 * Translates a touch event to a mouse event.
 * @param {Event} e
 */
function simulateMouseEvent(e) {
  e.preventDefault();
  var touch = e.changedTouches[0];
  var eventMap = {
    'touchstart': 'mousedown',
    'touchmove': 'mousemove',
    'touchend': 'mouseup'
  };
  touch.target.dispatchEvent(new MouseEvent(eventMap[e.type], {
    bubbles: true,
    cancelable: true,
    view: window,
    clientX: touch.clientX,
    clientY: touch.clientY,
    screenX: touch.screenX,
    screenY: touch.screenY
  }));
}

/**
 * Define a list of handles to create.
 *
 * @property {Array} position - The x and y ratio position of the handle within
 *      the crop region. Accepts a value between 0 to 1 in the order of [X, Y].
 * @property {Array} constraints - Define the side of the crop region that is to
 *      be affected by this handle. Accepts a value of 0 or 1 in the order of
 *      [TOP, RIGHT, BOTTOM, LEFT].
 * @property {String} cursor - The CSS cursor of this handle.
 */
var HANDLES = [{ position: [0.0, 0.0], constraints: [1, 0, 0, 1], cursor: 'nw-resize' }, { position: [0.5, 0.0], constraints: [1, 0, 0, 0], cursor: 'n-resize' }, { position: [1.0, 0.0], constraints: [1, 1, 0, 0], cursor: 'ne-resize' }, { position: [1.0, 0.5], constraints: [0, 1, 0, 0], cursor: 'e-resize' }, { position: [1.0, 1.0], constraints: [0, 1, 1, 0], cursor: 'se-resize' }, { position: [0.5, 1.0], constraints: [0, 0, 1, 0], cursor: 's-resize' }, { position: [0.0, 1.0], constraints: [0, 0, 1, 1], cursor: 'sw-resize' }, { position: [0.0, 0.5], constraints: [0, 0, 0, 1], cursor: 'w-resize' }];
var CropprCore = function () {
  function CropprCore(element, options) {
    var _this = this;
    var deferred = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;
    classCallCheck(this, CropprCore);
    this.options = CropprCore.parseOptions(options || {});
    if (!element.nodeName) {
      element = document.querySelector(element);
      if (element == null) {
        throw 'Unable to find element.';
      }
    }
    if (!element.getAttribute('src')) {
      throw 'Image src not provided.';
    }
    this._initialized = false;
    this._restore = {
      parent: element.parentNode,
      element: element
    };
    if (!deferred) {
      if (element.width === 0 || element.height === 0) {
        element.onload = function () {
          _this.initialize(element);
        };
      } else {
        this.initialize(element);
      }
    }
  }
  createClass(CropprCore, [{
    key: 'initialize',
    value: function initialize(element) {
      this.createDOM(element);
      this.options.convertToPixels(this.cropperEl);
      this.attachHandlerEvents();
      this.attachRegionEvents();
      this.attachOverlayEvents();
      this.box = this.initializeBox(this.options);
      this.redraw();
      this._initialized = true;
      if (this.options.onInitialize !== null) {
        this.options.onInitialize(this);
      }
    }
  }, {
    key: 'createDOM',
    value: function createDOM(targetEl) {
      this.containerEl = document.createElement('div');
      this.containerEl.className = 'croppr-container';
      this.eventBus = this.containerEl;
      enableTouch(this.containerEl);
      this.cropperEl = document.createElement('div');
      this.cropperEl.className = 'croppr';
      this.imageEl = document.createElement('img');
      this.imageEl.setAttribute('src', targetEl.getAttribute('src'));
      this.imageEl.setAttribute('alt', targetEl.getAttribute('alt'));
      this.imageEl.className = 'croppr-image';
      this.imageClippedEl = this.imageEl.cloneNode();
      this.imageClippedEl.className = 'croppr-imageClipped';
      this.regionEl = document.createElement('div');
      this.regionEl.className = 'croppr-region';
      this.overlayEl = document.createElement('div');
      this.overlayEl.className = 'croppr-overlay';
      var handleContainerEl = document.createElement('div');
      handleContainerEl.className = 'croppr-handleContainer';
      this.handles = [];
      for (var i = 0; i < HANDLES.length; i++) {
        var handle = new Handle(HANDLES[i].position, HANDLES[i].constraints, HANDLES[i].cursor, this.eventBus);
        this.handles.push(handle);
        handleContainerEl.appendChild(handle.el);
      }
      this.cropperEl.appendChild(this.imageEl);
      this.cropperEl.appendChild(this.imageClippedEl);
      this.cropperEl.appendChild(this.regionEl);
      this.cropperEl.appendChild(this.overlayEl);
      this.cropperEl.appendChild(handleContainerEl);
      this.containerEl.appendChild(this.cropperEl);
      targetEl.parentElement.replaceChild(this.containerEl, targetEl);
    }
    /**
     * Changes the image src.
     * @param {String} src
     */
  }, {
    key: 'setImage',
    value: function setImage(src) {
      var _this2 = this;
      this.imageEl.onload = function () {
        _this2.box = _this2.initializeBox(_this2.options);
        _this2.redraw();
      };
      this.imageEl.src = src;
      this.imageClippedEl.src = src;
      return this;
    }
  }, {
    key: 'destroy',
    value: function destroy() {
      this._restore.parent.replaceChild(this._restore.element, this.containerEl);
    }
    /**
     * Create a new box region with a set of options.
     * @param {Object} opts The options.
     * @returns {Box}
     */
  }, {
    key: 'initializeBox',
    value: function initializeBox(opts) {
      var width = opts.startSize.width;
      var height = opts.startSize.height;
      var box = new Box(0, 0, width, height);
      box.constrainToRatio(opts.aspectRatio, [0.5, 0.5]);
      var min = opts.minSize;
      var max = opts.maxSize;
      box.constrainToSize(max.width, max.height, min.width, min.height, [0.5, 0.5], opts.aspectRatio);
      var parentWidth = this.cropperEl.offsetWidth;
      var parentHeight = this.cropperEl.offsetHeight;
      box.constrainToBoundary(parentWidth, parentHeight, [0.5, 0.5]);
      var x = this.cropperEl.offsetWidth / 2 - box.width() / 2;
      var y = this.cropperEl.offsetHeight / 2 - box.height() / 2;
      box.move(x, y);
      return box;
    }
  }, {
    key: 'redraw',
    value: function redraw() {
      var _this3 = this;
      var width = Math.round(this.box.width()),
          height = Math.round(this.box.height()),
          x1 = Math.round(this.box.x1),
          y1 = Math.round(this.box.y1),
          x2 = Math.round(this.box.x2),
          y2 = Math.round(this.box.y2);
      window.requestAnimationFrame(function () {
        _this3.regionEl.style.transform = 'translate(' + x1 + 'px, ' + y1 + 'px)';
        _this3.regionEl.style.width = width + 'px';
        _this3.regionEl.style.height = height + 'px';
        _this3.imageClippedEl.style.clip = 'rect(' + y1 + 'px, ' + x2 + 'px, ' + y2 + 'px, ' + x1 + 'px)';
        var center = _this3.box.getAbsolutePoint([.5, .5]);
        var xSign = center[0] - _this3.cropperEl.offsetWidth / 2 >> 31;
        var ySign = center[1] - _this3.cropperEl.offsetHeight / 2 >> 31;
        var quadrant = (xSign ^ ySign) + ySign + ySign + 4;
        var foregroundHandleIndex = -2 * quadrant + 8;
        for (var i = 0; i < _this3.handles.length; i++) {
          var handle = _this3.handles[i];
          var handleWidth = handle.el.offsetWidth;
          var handleHeight = handle.el.offsetHeight;
          var left = x1 + width * handle.position[0] - handleWidth / 2;
          var top = y1 + height * handle.position[1] - handleHeight / 2;
          handle.el.style.transform = 'translate(' + Math.round(left) + 'px, ' + Math.round(top) + 'px)';
          handle.el.style.zIndex = foregroundHandleIndex == i ? 5 : 4;
        }
      });
    }
  }, {
    key: 'attachHandlerEvents',
    value: function attachHandlerEvents() {
      var eventBus = this.eventBus;
      eventBus.addEventListener('handlestart', this.onHandleMoveStart.bind(this));
      eventBus.addEventListener('handlemove', this.onHandleMoveMoving.bind(this));
      eventBus.addEventListener('handleend', this.onHandleMoveEnd.bind(this));
    }
  }, {
    key: 'attachRegionEvents',
    value: function attachRegionEvents() {
      var eventBus = this.eventBus;
      var self = this;
      this.regionEl.addEventListener('mousedown', onMouseDown);
      eventBus.addEventListener('regionstart', this.onRegionMoveStart.bind(this));
      eventBus.addEventListener('regionmove', this.onRegionMoveMoving.bind(this));
      eventBus.addEventListener('regionend', this.onRegionMoveEnd.bind(this));
      function onMouseDown(e) {
        e.stopPropagation();
        document.addEventListener('mouseup', onMouseUp);
        document.addEventListener('mousemove', onMouseMove);
        eventBus.dispatchEvent(new CustomEvent('regionstart', {
          detail: { mouseX: e.clientX, mouseY: e.clientY }
        }));
      }
      function onMouseMove(e) {
        e.stopPropagation();
        eventBus.dispatchEvent(new CustomEvent('regionmove', {
          detail: { mouseX: e.clientX, mouseY: e.clientY }
        }));
      }
      function onMouseUp(e) {
        e.stopPropagation();
        document.removeEventListener('mouseup', onMouseUp);
        document.removeEventListener('mousemove', onMouseMove);
        eventBus.dispatchEvent(new CustomEvent('regionend', {
          detail: { mouseX: e.clientX, mouseY: e.clientY }
        }));
      }
    }
  }, {
    key: 'attachOverlayEvents',
    value: function attachOverlayEvents() {
      var SOUTHEAST_HANDLE_IDX = 4;
      var self = this;
      var tmpBox = null;
      this.overlayEl.addEventListener('mousedown', onMouseDown);
      function onMouseDown(e) {
        e.stopPropagation();
        document.addEventListener('mouseup', onMouseUp);
        document.addEventListener('mousemove', onMouseMove);
        var container = self.cropperEl.getBoundingClientRect();
        var mouseX = e.clientX - container.left;
        var mouseY = e.clientY - container.top;
        tmpBox = self.box;
        self.box = new Box(mouseX, mouseY, mouseX + 1, mouseY + 1);
        self.eventBus.dispatchEvent(new CustomEvent('handlestart', {
          detail: { handle: self.handles[SOUTHEAST_HANDLE_IDX] }
        }));
      }
      function onMouseMove(e) {
        e.stopPropagation();
        self.eventBus.dispatchEvent(new CustomEvent('handlemove', {
          detail: { mouseX: e.clientX, mouseY: e.clientY }
        }));
      }
      function onMouseUp(e) {
        e.stopPropagation();
        document.removeEventListener('mouseup', onMouseUp);
        document.removeEventListener('mousemove', onMouseMove);
        if (self.box.width() === 1 && self.box.height() === 1) {
          self.box = tmpBox;
          return;
        }
        self.eventBus.dispatchEvent(new CustomEvent('handleend', {
          detail: { mouseX: e.clientX, mouseY: e.clientY }
        }));
      }
    }
  }, {
    key: 'onHandleMoveStart',
    value: function onHandleMoveStart(e) {
      var handle = e.detail.handle;
      var originPoint = [1 - handle.position[0], 1 - handle.position[1]];
      var _box$getAbsolutePoint = this.box.getAbsolutePoint(originPoint),
          _box$getAbsolutePoint2 = slicedToArray(_box$getAbsolutePoint, 2),
          originX = _box$getAbsolutePoint2[0],
          originY = _box$getAbsolutePoint2[1];
      this.activeHandle = { handle: handle, originPoint: originPoint, originX: originX, originY: originY };
      if (this.options.onCropStart !== null) {
        this.options.onCropStart(this.getValue());
      }
    }
  }, {
    key: 'onHandleMoveMoving',
    value: function onHandleMoveMoving(e) {
      var _e$detail = e.detail,
          mouseX = _e$detail.mouseX,
          mouseY = _e$detail.mouseY;
      var container = this.cropperEl.getBoundingClientRect();
      mouseX = mouseX - container.left;
      mouseY = mouseY - container.top;
      if (mouseX < 0) {
        mouseX = 0;
      } else if (mouseX > container.width) {
        mouseX = container.width;
      }
      if (mouseY < 0) {
        mouseY = 0;
      } else if (mouseY > container.height) {
        mouseY = container.height;
      }
      var origin = this.activeHandle.originPoint.slice();
      var originX = this.activeHandle.originX;
      var originY = this.activeHandle.originY;
      var handle = this.activeHandle.handle;
      var TOP_MOVABLE = handle.constraints[0] === 1;
      var RIGHT_MOVABLE = handle.constraints[1] === 1;
      var BOTTOM_MOVABLE = handle.constraints[2] === 1;
      var LEFT_MOVABLE = handle.constraints[3] === 1;
      var MULTI_AXIS = (LEFT_MOVABLE || RIGHT_MOVABLE) && (TOP_MOVABLE || BOTTOM_MOVABLE);
      var x1 = LEFT_MOVABLE || RIGHT_MOVABLE ? originX : this.box.x1;
      var x2 = LEFT_MOVABLE || RIGHT_MOVABLE ? originX : this.box.x2;
      var y1 = TOP_MOVABLE || BOTTOM_MOVABLE ? originY : this.box.y1;
      var y2 = TOP_MOVABLE || BOTTOM_MOVABLE ? originY : this.box.y2;
      x1 = LEFT_MOVABLE ? mouseX : x1;
      x2 = RIGHT_MOVABLE ? mouseX : x2;
      y1 = TOP_MOVABLE ? mouseY : y1;
      y2 = BOTTOM_MOVABLE ? mouseY : y2;
      var isFlippedX = false,
          isFlippedY = false;
      if (LEFT_MOVABLE || RIGHT_MOVABLE) {
        isFlippedX = LEFT_MOVABLE ? mouseX > originX : mouseX < originX;
      }
      if (TOP_MOVABLE || BOTTOM_MOVABLE) {
        isFlippedY = TOP_MOVABLE ? mouseY > originY : mouseY < originY;
      }
      if (isFlippedX) {
        var tmp = x1;x1 = x2;x2 = tmp;
        origin[0] = 1 - origin[0];
      }
      if (isFlippedY) {
        var _tmp = y1;y1 = y2;y2 = _tmp;
        origin[1] = 1 - origin[1];
      }
      var box = new Box(x1, y1, x2, y2);
      if (this.options.aspectRatio) {
        var ratio = this.options.aspectRatio;
        var isVerticalMovement = false;
        if (MULTI_AXIS) {
          isVerticalMovement = mouseY > box.y1 + ratio * box.width() || mouseY < box.y2 - ratio * box.width();
        } else if (TOP_MOVABLE || BOTTOM_MOVABLE) {
          isVerticalMovement = true;
        }
        var ratioMode = isVerticalMovement ? 'width' : 'height';
        box.constrainToRatio(ratio, origin, ratioMode);
      }
      var min = this.options.minSize;
      var max = this.options.maxSize;
      box.constrainToSize(max.width, max.height, min.width, min.height, origin, this.options.aspectRatio);
      var parentWidth = this.cropperEl.offsetWidth;
      var parentHeight = this.cropperEl.offsetHeight;
      box.constrainToBoundary(parentWidth, parentHeight, origin);
      this.box = box;
      this.redraw();
      if (this.options.onCropMove !== null) {
        this.options.onCropMove(this.getValue());
      }
    }
  }, {
    key: 'onHandleMoveEnd',
    value: function onHandleMoveEnd(e) {
      if (this.options.onCropEnd !== null) {
        this.options.onCropEnd(this.getValue());
      }
    }
  }, {
    key: 'onRegionMoveStart',
    value: function onRegionMoveStart(e) {
      var _e$detail2 = e.detail,
          mouseX = _e$detail2.mouseX,
          mouseY = _e$detail2.mouseY;
      var container = this.cropperEl.getBoundingClientRect();
      mouseX = mouseX - container.left;
      mouseY = mouseY - container.top;
      this.currentMove = {
        offsetX: mouseX - this.box.x1,
        offsetY: mouseY - this.box.y1
      };
      if (this.options.onCropStart !== null) {
        this.options.onCropStart(this.getValue());
      }
    }
  }, {
    key: 'onRegionMoveMoving',
    value: function onRegionMoveMoving(e) {
      var _e$detail3 = e.detail,
          mouseX = _e$detail3.mouseX,
          mouseY = _e$detail3.mouseY;
      var _currentMove = this.currentMove,
          offsetX = _currentMove.offsetX,
          offsetY = _currentMove.offsetY;
      var container = this.cropperEl.getBoundingClientRect();
      mouseX = mouseX - container.left;
      mouseY = mouseY - container.top;
      this.box.move(mouseX - offsetX, mouseY - offsetY);
      if (this.box.x1 < 0) {
        this.box.move(0, null);
      }
      if (this.box.x2 > container.width) {
        this.box.move(container.width - this.box.width(), null);
      }
      if (this.box.y1 < 0) {
        this.box.move(null, 0);
      }
      if (this.box.y2 > container.height) {
        this.box.move(null, container.height - this.box.height());
      }
      this.redraw();
      if (this.options.onCropMove !== null) {
        this.options.onCropMove(this.getValue());
      }
    }
  }, {
    key: 'onRegionMoveEnd',
    value: function onRegionMoveEnd(e) {
      if (this.options.onCropEnd !== null) {
        this.options.onCropEnd(this.getValue());
      }
    }
  }, {
    key: 'getValue',
    value: function getValue() {
      var mode = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : null;
      if (mode === null) {
        mode = this.options.returnMode;
      }
      if (mode == 'real') {
        var actualWidth = this.imageEl.naturalWidth;
        var actualHeight = this.imageEl.naturalHeight;
        var _imageEl$getBoundingC = this.imageEl.getBoundingClientRect(),
            elementWidth = _imageEl$getBoundingC.width,
            elementHeight = _imageEl$getBoundingC.height;
        var factorX = actualWidth / elementWidth;
        var factorY = actualHeight / elementHeight;
        return {
          x: Math.round(this.box.x1 * factorX),
          y: Math.round(this.box.y1 * factorY),
          width: Math.round(this.box.width() * factorX),
          height: Math.round(this.box.height() * factorY)
        };
      } else if (mode == 'ratio') {
        var _imageEl$getBoundingC2 = this.imageEl.getBoundingClientRect(),
            _elementWidth = _imageEl$getBoundingC2.width,
            _elementHeight = _imageEl$getBoundingC2.height;
        return {
          x: round(this.box.x1 / _elementWidth, 3),
          y: round(this.box.y1 / _elementHeight, 3),
          width: round(this.box.width() / _elementWidth, 3),
          height: round(this.box.height() / _elementHeight, 3)
        };
      } else if (mode == 'raw') {
        return {
          x: Math.round(this.box.x1),
          y: Math.round(this.box.y1),
          width: Math.round(this.box.width()),
          height: Math.round(this.box.height())
        };
      }
    }
  }], [{
    key: 'parseOptions',
    value: function parseOptions(opts) {
      var defaults$$1 = {
        aspectRatio: null,
        maxSize: { width: null, height: null },
        minSize: { width: null, height: null },
        startSize: { width: 100, height: 100, unit: '%' },
        returnMode: 'real',
        onInitialize: null,
        onCropStart: null,
        onCropMove: null,
        onCropEnd: null
      };
      var aspectRatio = null;
      if (opts.aspectRatio !== undefined) {
        if (typeof opts.aspectRatio === 'number') {
          aspectRatio = opts.aspectRatio;
        } else if (opts.aspectRatio instanceof Array) {
          aspectRatio = opts.aspectRatio[1] / opts.aspectRatio[0];
        }
      }
      var maxSize = null;
      if (opts.maxSize !== undefined && opts.maxSize !== null) {
        maxSize = {
          width: opts.maxSize[0] || null,
          height: opts.maxSize[1] || null,
          unit: opts.maxSize[2] || 'px'
        };
      }
      var minSize = null;
      if (opts.minSize !== undefined && opts.minSize !== null) {
        minSize = {
          width: opts.minSize[0] || null,
          height: opts.minSize[1] || null,
          unit: opts.minSize[2] || 'px'
        };
      }
      var startSize = null;
      if (opts.startSize !== undefined && opts.startSize !== null) {
        startSize = {
          width: opts.startSize[0] || null,
          height: opts.startSize[1] || null,
          unit: opts.startSize[2] || '%'
        };
      }
      var onInitialize = null;
      if (typeof opts.onInitialize === 'function') {
        onInitialize = opts.onInitialize;
      }
      var onCropStart = null;
      if (typeof opts.onCropStart === 'function') {
        onCropStart = opts.onCropStart;
      }
      var onCropEnd = null;
      if (typeof opts.onCropEnd === 'function') {
        onCropEnd = opts.onCropEnd;
      }
      var onCropMove = null;
      if (typeof opts.onUpdate === 'function') {
        console.warn('Croppr.js: `onUpdate` is deprecated and will be removed in the next major release. Please use `onCropMove` or `onCropEnd` instead.');
        onCropMove = opts.onUpdate;
      }
      if (typeof opts.onCropMove === 'function') {
        onCropMove = opts.onCropMove;
      }
      var returnMode = null;
      if (opts.returnMode !== undefined) {
        var s = opts.returnMode.toLowerCase();
        if (['real', 'ratio', 'raw'].indexOf(s) === -1) {
          throw "Invalid return mode.";
        }
        returnMode = s;
      }
      var convertToPixels = function convertToPixels(container) {
        var width = container.offsetWidth;
        var height = container.offsetHeight;
        var sizeKeys = ['maxSize', 'minSize', 'startSize'];
        for (var i = 0; i < sizeKeys.length; i++) {
          var key = sizeKeys[i];
          if (this[key] !== null) {
            if (this[key].unit == '%') {
              if (this[key].width !== null) {
                this[key].width = this[key].width / 100 * width;
              }
              if (this[key].height !== null) {
                this[key].height = this[key].height / 100 * height;
              }
            }
            delete this[key].unit;
          }
        }
      };
      var defaultValue = function defaultValue(v, d) {
        return v !== null ? v : d;
      };
      return {
        aspectRatio: defaultValue(aspectRatio, defaults$$1.aspectRatio),
        maxSize: defaultValue(maxSize, defaults$$1.maxSize),
        minSize: defaultValue(minSize, defaults$$1.minSize),
        startSize: defaultValue(startSize, defaults$$1.startSize),
        returnMode: defaultValue(returnMode, defaults$$1.returnMode),
        onInitialize: defaultValue(onInitialize, defaults$$1.onInitialize),
        onCropStart: defaultValue(onCropStart, defaults$$1.onCropStart),
        onCropMove: defaultValue(onCropMove, defaults$$1.onCropMove),
        onCropEnd: defaultValue(onCropEnd, defaults$$1.onCropEnd),
        convertToPixels: convertToPixels
      };
    }
  }]);
  return CropprCore;
}();
function round(value, decimals) {
  return Number(Math.round(value + 'e' + decimals) + 'e-' + decimals);
}

var Croppr$1 = function (_CropprCore) {
  inherits(Croppr, _CropprCore);
  /**
   * @constructor
   * Calls the CropprCore's constructor.
   */
  function Croppr(element, options) {
    var _deferred = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;
    classCallCheck(this, Croppr);
    return possibleConstructorReturn(this, (Croppr.__proto__ || Object.getPrototypeOf(Croppr)).call(this, element, options, _deferred));
  }
  /**
   * Gets the value of the crop region.
   * @param {String} [mode] Which mode of calculation to use: 'real', 'ratio' or
   *      'raw'.
   */
  createClass(Croppr, [{
    key: 'getValue',
    value: function getValue(mode) {
      return get(Croppr.prototype.__proto__ || Object.getPrototypeOf(Croppr.prototype), 'getValue', this).call(this, mode);
    }
    /**
     * Changes the image src.
     * @param {String} src
     */
  }, {
    key: 'setImage',
    value: function setImage(src) {
      return get(Croppr.prototype.__proto__ || Object.getPrototypeOf(Croppr.prototype), 'setImage', this).call(this, src);
    }
  }, {
    key: 'destroy',
    value: function destroy() {
      return get(Croppr.prototype.__proto__ || Object.getPrototypeOf(Croppr.prototype), 'destroy', this).call(this);
    }
    /**
     * Moves the crop region to a specified coordinate.
     * @param {Number} x
     * @param {Number} y
     */
  }, {
    key: 'moveTo',
    value: function moveTo(x, y) {
      this.box.move(x, y);
      this.redraw();
      if (this.options.onCropEnd !== null) {
        this.options.onCropEnd(this.getValue());
      }
      return this;
    }
    /**
     * Resizes the crop region to a specified width and height.
     * @param {Number} width
     * @param {Number} height
     * @param {Array} origin The origin point to resize from.
     *      Defaults to [0.5, 0.5] (center).
     */
  }, {
    key: 'resizeTo',
    value: function resizeTo(width, height) {
      var origin = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : [.5, .5];
      this.box.resize(width, height, origin);
      this.redraw();
      if (this.options.onCropEnd !== null) {
        this.options.onCropEnd(this.getValue());
      }
      return this;
    }
    /**
     * Scale the crop region by a factor.
     * @param {Number} factor
     * @param {Array} origin The origin point to resize from.
     *      Defaults to [0.5, 0.5] (center).
     */
  }, {
    key: 'scaleBy',
    value: function scaleBy(factor) {
      var origin = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : [.5, .5];
      this.box.scale(factor, origin);
      this.redraw();
      if (this.options.onCropEnd !== null) {
        this.options.onCropEnd(this.getValue());
      }
      return this;
    }
  }, {
    key: 'reset',
    value: function reset() {
      this.box = this.initializeBox(this.options);
      this.redraw();
      if (this.options.onCropEnd !== null) {
        this.options.onCropEnd(this.getValue());
      }
      return this;
    }
  }]);
  return Croppr;
}(CropprCore);

return Croppr$1;

})));
