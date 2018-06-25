Chapter
=======

A chapter is always followed by aline with =

Section
*******

A section is always followed by aline with *

SubSection
^^^^^^^^^^

A subsection is always followed by aline with ^

SubSubSection
_____________

A subsubsection is always followed by aline with _

Using *italic*  

Using **bold**  

Math formula

.. math::

  - \nabla^2 u = f

or :math:`x \in \mathbb{R}`

including a python code

.. code-block:: python

  from numpy import zeros

.. note:: If you have pandoc and pandocfilters installed, you can convert directly a jupyter notebook into an rst file, that can be included like this introduction.rst file.

.. note:: you can also change the html theme, or include latex macros (see my pyccel documentation for example)

.. note:: whenever you push, the online documentation will be automatically updated on the  `url <http://mlhiphy.readthedocs.io/en/latest/>`_

.. note:: more information on rst files can be found `here <http://docutils.sourceforge.net/docs/user/rst/quickref.html>`_ 

.. note:: you can change the settings of this documentation using the **conf.py** file



